# Lint as: python3
"""Core GraphNet architecture and PressNet specific model."""

import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf
from src import common

# --- Graph Definitions ---
EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])

# --- Normalization Module ---
class Normalizer(snt.AbstractModule):
  def __init__(self, size, name='Normalizer'):
    super(Normalizer, self).__init__(name=name)
    self._size = size
    with self._enter_variable_scope():
      self._max_accumulations = 10**6
      self._acc_count = tf.Variable(0, dtype=tf.float32, trainable=False, name='acc_count')
      self._num_accumulations = tf.Variable(0, dtype=tf.float32, trainable=False, name='num_accumulations')
      self._mean_sum = tf.Variable(tf.zeros(size, tf.float32), trainable=False, name='mean_sum')
      self._std_sum = tf.Variable(tf.zeros(size, tf.float32), trainable=False, name='std_sum')
      self._acc_sum = tf.Variable(tf.zeros(size, tf.float32), trainable=False, name='acc_sum')
      self._acc_sum_squared = tf.Variable(tf.zeros(size, tf.float32), trainable=False, name='acc_sum_squared')

  def _build(self, data, accumulate=True):
    if accumulate:
      # Update running stats
      count = tf.to_float(tf.shape(data)[0])
      data_sum = tf.reduce_sum(data, axis=0)
      squared_data_sum = tf.reduce_sum(data**2, axis=0)
      
      update_op = tf.group(
          tf.assign_add(self._acc_sum, data_sum),
          tf.assign_add(self._acc_sum_squared, squared_data_sum),
          tf.assign_add(self._acc_count, count),
          tf.assign_add(self._num_accumulations, 1.0)
      )
      with tf.control_dependencies([update_op]):
        return (data - self._mean()) / self._std_with_epsilon()
    return (data - self._mean()) / self._std_with_epsilon()

  def _mean(self):
    safe_count = tf.maximum(self._acc_count, 1e-5)
    return self._acc_sum / safe_count

  def _std_with_epsilon(self):
    safe_count = tf.maximum(self._acc_count, 1e-5)
    std = tf.sqrt(tf.maximum(0.0, self._acc_sum_squared / safe_count - self._mean()**2))
    return tf.maximum(std, 1e-8)
    
  def inverse(self, normalized_data):
    return normalized_data * self._std_with_epsilon() + self._mean()

# --- Core GNN Architecture ---
class GraphNetBlock(snt.AbstractModule):
  def __init__(self, model_fn, name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self._model_fn = model_fn

  def _update_edge_features(self, node_features, edge_set):
    senders = edge_set.senders
    receivers = edge_set.receivers
    sender_features = tf.gather(node_features, senders)
    receiver_features = tf.gather(node_features, receivers)
    features = [sender_features, receiver_features, edge_set.features]
    return self._model_fn()(tf.concat(features, axis=-1))

  def _update_node_features(self, node_features, edge_sets):
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]
    for edge_set in edge_sets:
      aggregated = tf.math.unsorted_segment_sum(edge_set.features, edge_set.receivers, num_nodes)
      features.append(aggregated)
    return self._model_fn()(tf.concat(features, axis=-1))

  def _build(self, graph):
    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated = self._update_edge_features(graph.node_features, edge_set)
      new_edge_sets.append(edge_set._replace(features=updated))

    new_node_features = self._update_node_features(graph.node_features, new_edge_sets)
    
    # Residual Connections
    new_node_features += graph.node_features
    new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)

class EncodeProcessDecode(snt.AbstractModule):
  def __init__(self, output_size, latent_size, num_layers, message_passing_steps, name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps

  def _make_mlp(self, output_size, layer_norm=True):
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _build(self, graph):
    # Encoder
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size)(graph.node_features)
      new_edges_sets = []
      for edge_set in graph.edge_sets:
        latent = self._make_mlp(self._latent_size)(edge_set.features)
        new_edges_sets.append(edge_set._replace(features=latent))
      latent_graph = MultiGraph(node_latents, new_edges_sets)

    # Processor
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    for _ in range(self._message_passing_steps):
      latent_graph = GraphNetBlock(model_fn)(latent_graph)

    # Decoder
    with tf.variable_scope('decoder'):
      return self._make_mlp(self._output_size, layer_norm=False)(latent_graph.node_features)

# --- PressNet Implementation ---
class PressNetModel(snt.AbstractModule):
  def __init__(self, learned_model, name='PressNetModel'):
    super(PressNetModel, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model
      self._output_normalizer = Normalizer(size=3, name='output_normalizer')
      self._edge_normalizer = Normalizer(size=8, name='edge_normalizer')
      self._world_edge_normalizer = Normalizer(size=4, name='world_edge_normalizer')
      self._node_normalizer = Normalizer(size=5, name='node_normalizer')

  def _build_graph(self, inputs, is_training):
    # 1. Node Features
    node_type = tf.reshape(inputs['node_type'], [-1])
    type_idx = tf.cast(tf.not_equal(node_type, common.NodeType.NORMAL), tf.int32)
    node_type_one_hot = tf.one_hot(type_idx, 2)

    world_pos = tf.reshape(inputs['world_pos'], [-1, 3])
    prev_pos = tf.reshape(inputs['prev|world_pos'], [-1, 3])
    target_pos = tf.reshape(inputs['target|world_pos'], [-1, 3])

    velocity_metal = world_pos - prev_pos
    velocity_tool = target_pos - world_pos
    is_tool = tf.equal(node_type, common.NodeType.OBSTACLE)
    velocity = tf.where(tf.tile(tf.expand_dims(is_tool, -1), [1, 3]), velocity_tool, velocity_metal)
    
    node_features = tf.concat([node_type_one_hot, velocity], axis=-1)

    # 2. Mesh Edges
    cells = tf.cast(tf.reshape(inputs['cells'], [-1, 3]), tf.int32)
    senders, receivers = common.triangles_to_edges(cells)
    mesh_pos = tf.reshape(inputs['mesh_pos'], [-1, 3])
    
    rel_world = (tf.gather(world_pos, senders) - tf.gather(world_pos, receivers))
    rel_mesh = (tf.gather(mesh_pos, senders) - tf.gather(mesh_pos, receivers))
    
    edge_features = tf.concat([
        rel_world, tf.norm(rel_world, axis=-1, keepdims=True),
        rel_mesh, tf.norm(rel_mesh, axis=-1, keepdims=True)
    ], axis=-1)
    edge_features = self._edge_normalizer(edge_features, is_training)

    # 3. World Edges
    w_senders, w_receivers = common.construct_world_edges(world_pos, inputs['node_type'])
    rel_world_w = (tf.gather(world_pos, w_senders) - tf.gather(world_pos, w_receivers))
    
    world_edge_features = tf.concat([
        rel_world_w, tf.norm(rel_world_w, axis=-1, keepdims=True)
    ], axis=-1)
    world_edge_features = self._world_edge_normalizer(world_edge_features, is_training)

    # Explicit Shapes
    node_features.set_shape([None, 5])
    edge_features.set_shape([None, 8])
    world_edge_features.set_shape([None, 4])

    return MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[
            EdgeSet(name='mesh_edges', features=edge_features, receivers=receivers, senders=senders),
            EdgeSet(name='world_edges', features=world_edge_features, receivers=w_receivers, senders=w_senders)
        ])

  def loss(self, inputs):
    graph = self._build_graph(inputs, is_training=True)
    pred_norm = self._learned_model(graph)
    
    target_disp = tf.reshape(inputs['target|world_pos'] - inputs['world_pos'], [-1, 3])
    target_norm = self._output_normalizer(target_disp, accumulate=True)
    
    error = tf.reduce_sum((target_norm - pred_norm)**2, axis=1)
    mask = tf.equal(tf.reshape(inputs['node_type'], [-1]), common.NodeType.NORMAL)
    return tf.reduce_mean(error[mask])

  def _build(self, inputs, is_training=False):
    graph = self._build_graph(inputs, is_training)
    pred_norm = self._learned_model(graph)
    pred_disp = self._output_normalizer.inverse(pred_norm)
    return tf.reshape(inputs['world_pos'], [-1, 3]) + pred_disp