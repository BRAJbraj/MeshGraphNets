# Lint as: python3
"""Evaluation/Rollout logic for PressNet."""

import tensorflow.compat.v1 as tf
from src import common

def evaluate(model, inputs):
  """Performs rollout evaluation."""
  
  # Ground Truth Trajectory [Time, Nodes, 3]
  full_trajectory = inputs['world_pos'] 
  num_time_steps = tf.shape(full_trajectory)[0]
  num_rollout_steps = num_time_steps - 1

  # Initial State
  current_pos = full_trajectory[0]
  prev_pos = full_trajectory[0] # Assume 0 velocity at start
  
  # Static features (Shape invariants)
  node_type = inputs['node_type'][0]
  cells = inputs['cells'][0]
  mesh_pos = inputs['mesh_pos'][0]

  def step_fn(step, current_pos, prev_pos, trajectory_array):
    # Predict next position
    prediction_inputs = {
        'world_pos': current_pos,
        'prev|world_pos': prev_pos,
        'node_type': node_type,
        'cells': cells,
        'mesh_pos': mesh_pos,
        # Target position for the Tool (Kinematic) comes from Ground Truth
        'target|world_pos': full_trajectory[step + 1] 
    }
    
    # The model predicts the NEXT position
    next_pos = model._build(prediction_inputs, is_training=False)

    # Kinematic Enforcement: 
    # For Tool nodes (Obstacles), we force them to follow the Ground Truth
    # because the GNN only predicts the Metal (Deformable) nodes.
    mask = tf.equal(node_type, common.NodeType.NORMAL)[:, 0]
    next_pos = tf.where(mask, next_pos, full_trajectory[step + 1])

    # Store result
    trajectory_array = trajectory_array.write(step, next_pos)

    # Update: Current becomes Previous for the next step
    return step + 1, next_pos, current_pos, trajectory_array

  # Execute Rollout Loop
  step = tf.constant(0)
  trajectory_ta = tf.TensorArray(dtype=tf.float32, size=num_rollout_steps)

  _, _, _, final_ta = tf.while_loop(
      cond=lambda s, c, p, t: tf.less(s, num_rollout_steps),
      body=step_fn,
      loop_vars=(step, current_pos, prev_pos, trajectory_ta),
      parallel_iterations=1
  )

  return None, final_ta.stack() # Return (Scalars, Trajectory)