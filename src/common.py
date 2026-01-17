# Lint as: python3
"""Common data structures and helper functions."""

import enum
import tensorflow.compat.v1 as tf

class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9

def triangles_to_edges(faces):
  """Computes mesh edges from triangles."""
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))

def squared_dist(A, B):
  """Computes squared euclidean distance between A and B."""
  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])
  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])
  return row_norms_A - 2 * tf.matmul(A, B, False, True) + row_norms_B

def construct_world_edges(world_pos, node_type, radius=50):
    """Constructs edges between Tool (OBSTACLE) and Metal (NORMAL)."""
    deformable_idx = tf.where(tf.equal(node_type[:, 0], NodeType.NORMAL))
    actuator_idx = tf.where(tf.equal(node_type[:, 0], NodeType.OBSTACLE))

    B = tf.squeeze(tf.gather(world_pos, deformable_idx))
    A = tf.squeeze(tf.gather(world_pos, actuator_idx))
    
    # Cast for precision safety
    A = tf.cast(A, tf.float32)
    B = tf.cast(B, tf.float32)

    dists = squared_dist(A, B)
    rel_close_pair_idx = tf.where(tf.math.less(dists, radius ** 2))

    close_pair_actuator = tf.gather(actuator_idx, rel_close_pair_idx[:, 0])
    close_pair_def = tf.gather(deformable_idx, rel_close_pair_idx[:, 1])

    senders_a = tf.squeeze(close_pair_actuator)
    receivers_a = tf.squeeze(close_pair_def)
    senders_b = tf.squeeze(close_pair_def)
    receivers_b = tf.squeeze(close_pair_actuator)

    # Force Rank 1 to prevent concat errors on empty edges
    senders = tf.concat([tf.reshape(senders_a, [-1]), tf.reshape(senders_b, [-1])], axis=0)
    receivers = tf.concat([tf.reshape(receivers_a, [-1]), tf.reshape(receivers_b, [-1])], axis=0)

    return senders, receivers