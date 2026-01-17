# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Core learned graph net model."""

import collections
import functools
import sonnet as snt
import tensorflow.compat.v1 as tf

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


# --- HELPER 1: Safe Shape Access ---
def _get_last_dim_safe(tensor):
    """Safely gets the last dimension (feature size) or returns None."""
    if tensor.shape.ndims is None:
        return None
    return tensor.shape.as_list()[-1]

# --- HELPER 2: Shape Restoration ---
def _ensure_last_dim(tensor, dim):
    """Sets the last dimension of the tensor if it is known."""
    if dim is not None:
        shape = tensor.shape.as_list() if tensor.shape.ndims is not None else [None, None]
        # Set the last dimension to the known value 'dim'
        tensor.set_shape(shape[:-1] + [dim])
    return tensor

# --- HELPER 3: Rank Restoration (Fixes Gradient Error) ---
def _ensure_rank_1(tensor):
    """Hints that the tensor is 1-dimensional (vector) with unknown length.

    This tells TF: "I don't know the length (None), but I know it's a 1D vector."
    This fixes the 'None - None' TypeError in gradient calculations.
    """
    if tensor.shape.ndims is None:
        tensor.set_shape([None])
    return tensor
# -----------------------------------


class GraphNetBlock(snt.AbstractModule):
  """Multi-Edge Interaction Network with residual connections."""

  def __init__(self, model_fn, name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self._model_fn = model_fn

  def _update_edge_features(self, node_features, edge_set):
    """Aggregrates node features, and applies edge function."""

    # --- FIX 1: Ensure Indices are Rank 1 ---
    # We guarantee these are 1D vectors so TF can calculate gradients.
    senders = _ensure_rank_1(edge_set.senders)
    receivers = _ensure_rank_1(edge_set.receivers)
    # ----------------------------------------

    sender_features = tf.gather(node_features, senders)
    receiver_features = tf.gather(node_features, receivers)

    # --- FIX 2: Recover Lost Shapes for Edges ---
    latent_dim = _get_last_dim_safe(node_features)
    if latent_dim is None:
        latent_dim = _get_last_dim_safe(edge_set.features)

    if latent_dim is not None:
        sender_features = _ensure_last_dim(sender_features, latent_dim)
        receiver_features = _ensure_last_dim(receiver_features, latent_dim)
    # --------------------------------------------

    features = [sender_features, receiver_features, edge_set.features]

    with tf.variable_scope(edge_set.name+'_edge_fn'):
      concat_features = tf.concat(features, axis=-1)

      # --- FIX 3: Calculate Expected MLP Input Size ---
      expected_dim = 0
      for f in features:
          d = _get_last_dim_safe(f)
          if d is not None:
              expected_dim += d

      if expected_dim > 0:
          concat_features = _ensure_last_dim(concat_features, expected_dim)
      # ----------------------------------------------

      return self._model_fn()(concat_features)

  def _update_node_features(self, node_features, edge_sets):
    """Aggregrates edge features, and applies node function."""
    num_nodes = tf.shape(node_features)[0]
    features = [node_features]

    # --- FIX 4: Identify Latent Dimension ---
    latent_dim = _get_last_dim_safe(node_features)
    # ----------------------------------------

    for edge_set in edge_sets:
      # --- FIX 5: Ensure Receivers are Rank 1 ---
      receivers = _ensure_rank_1(edge_set.receivers)
      # ------------------------------------------

      aggregated_messages = tf.math.unsorted_segment_sum(edge_set.features,
                                                         receivers,
                                                         num_nodes)

      # --- FIX 6: Restore Shape After Aggregation ---
      if latent_dim is not None:
          aggregated_messages = _ensure_last_dim(aggregated_messages, latent_dim)
      # ----------------------------------------------

      features.append(aggregated_messages)

    with tf.variable_scope('node_fn'):
      concat_features = tf.concat(features, axis=-1)

      # --- FIX 7: Calculate Expected MLP Input Size for Nodes ---
      expected_dim = 0
      for f in features:
          d = _get_last_dim_safe(f)
          if d is not None:
              expected_dim += d
          elif latent_dim is not None:
              expected_dim += latent_dim

      if expected_dim > 0:
          concat_features = _ensure_last_dim(concat_features, expected_dim)
      # ------------------------------------------------------

      return self._model_fn()(concat_features)

  def _build(self, graph):
    """Applies GraphNetBlock and returns updated MultiGraph."""

    # apply edge functions
    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated_features = self._update_edge_features(graph.node_features,
                                                    edge_set)
      new_edge_sets.append(edge_set._replace(features=updated_features))

    # apply node function
    new_node_features = self._update_node_features(graph.node_features,
                                                   new_edge_sets)

    # add residual connections
    new_node_features += graph.node_features
    new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)


class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode GraphNet model."""

  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps

  def _make_mlp(self, output_size, layer_norm=True):
    """Builds an MLP."""
    widths = [self._latent_size] * self._num_layers + [output_size]
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _encoder(self, graph):
    """Encodes node and edge features into latent features."""
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size)(graph.node_features)
      new_edges_sets = []
      for edge_set in graph.edge_sets:
        latent = self._make_mlp(self._latent_size)(edge_set.features)
        new_edges_sets.append(edge_set._replace(features=latent))
    return MultiGraph(node_latents, new_edges_sets)

  def _decoder(self, graph):
    """Decodes node features from graph."""
    with tf.variable_scope('decoder'):
      decoder = self._make_mlp(self._output_size, layer_norm=False)
      return decoder(graph.node_features)

  def _build(self, graph):
    """Encodes and processes a multigraph, and returns node features."""
    model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
    latent_graph = self._encoder(graph)
    for _ in range(self._message_passing_steps):
      latent_graph = GraphNetBlock(model_fn)(latent_graph)
    return self._decoder(latent_graph)
