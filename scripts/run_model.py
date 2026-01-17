# Lint as: python3
"""Main execution script for PressNet Training and Evaluation."""

import sys
import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
from absl import app
from absl import flags
from absl import logging

# --- Path Setup to find 'src' ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import model
from src import dataset
from src import evaluator
from src import common

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Train model, or run evaluation.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None, 'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'], 'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of paths to use for eval rollouts.')
flags.DEFINE_integer('num_training_steps', int(2e6), 'No. of training steps.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')

# Config for PressNet
PARAMS = {
    'noise': 0.003,
    'gamma': 0.1,
    'field': 'world_pos',
    'history': True,
    'size': 3
}

def learner(learned_model):
  """Runs training loop."""
  pressnet_model = model.PressNetModel(learned_model)

  # 1. Load Data
  ds_train = dataset.load_dataset(FLAGS.dataset_dir, 'train')
  ds_train = dataset.add_targets(ds_train, [PARAMS['field']], add_history=PARAMS['history'])
  ds_train = dataset.split_and_preprocess(ds_train, PARAMS['field'], 
                                          noise_scale=PARAMS['noise'], noise_gamma=PARAMS['gamma'])
  inputs = tf.data.make_one_shot_iterator(dataset.batch_dataset(ds_train, FLAGS.batch_size)).get_next()

  # 2. Validation Data
  ds_val = dataset.load_dataset(FLAGS.dataset_dir, 'valid')
  ds_val = dataset.add_targets(ds_val, [PARAMS['field']], add_history=PARAMS['history'])
  ds_val = dataset.split_and_preprocess(ds_val, PARAMS['field'], 
                                        noise_scale=0.0, noise_gamma=PARAMS['gamma'])
  val_inputs = tf.data.make_one_shot_iterator(dataset.batch_dataset(ds_val, FLAGS.batch_size)).get_next()

  # 3. Loss Ops
  loss_op = pressnet_model.loss(inputs)
  val_loss_op = pressnet_model.loss(val_inputs)

  # 4. Optimizer
  global_step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(1e-4, global_step, decay_steps=5e6, decay_rate=0.1) + 1e-6
  train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op, global_step=global_step)

  # 5. Hooks & Savers
  hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)]
  best_saver = tf.train.Saver(max_to_keep=1, name='best_saver')

  best_val_loss = float('inf')
  patience = 10
  patience_count = 0

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      hooks=hooks,
      save_checkpoint_secs=None,
      save_checkpoint_steps=1000,
      save_summaries_steps=100) as sess:

    logging.info('🚀 Training Started...')
    while not sess.should_stop():
      _, step, loss = sess.run([train_op, global_step, loss_op])

      if step % 100 == 0:
        logging.info('Step %d: Loss = %.6f', step, loss)

      # Validation Check
      if step % 1000 == 0 and step > 0:
        try:
          val_loss = sess.run(val_loss_op)
          logging.info('Step %d: Validation Loss = %.6f', step, val_loss)

          if val_loss < best_val_loss:
            logging.info("✅ New best score! (Old: %.6f -> New: %.6f)", best_val_loss, val_loss)
            best_val_loss = val_loss
            patience_count = 0
            
            # Smart Session Unwrap to save Best Model
            raw_sess = sess
            while type(raw_sess).__name__ not in ['Session', 'InteractiveSession']:
                if hasattr(raw_sess, '_sess'): raw_sess = raw_sess._sess
                else: break
            
            logging.info("💾 Saving 'best_model'...")
            best_saver.save(raw_sess, os.path.join(FLAGS.checkpoint_dir, 'best_model'))
          else:
            patience_count += 1
            if patience_count >= patience:
              logging.info("🛑 EARLY STOPPING TRIGGERED.")
              break
        except tf.errors.OutOfRangeError:
          pass

def run_eval(learned_model):
  """Runs evaluation."""
  pressnet_model = model.PressNetModel(learned_model)
  
  ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
  ds = dataset.add_targets(ds, [PARAMS['field']], add_history=PARAMS['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()

  _, traj_ops = evaluator.evaluate(pressnet_model, inputs)

  with tf.train.MonitoredSession(
      session_creator=tf.train.ChiefSessionCreator(
          checkpoint_dir=FLAGS.checkpoint_dir,
          config=tf.ConfigProto(log_device_placement=False))) as sess:

    trajectories = []
    for i in range(FLAGS.num_rollouts):
      try:
        logging.info('Rolling out trajectory %d...', i + 1)
        traj = sess.run(traj_ops)
        trajectories.append(traj)
      except tf.errors.OutOfRangeError:
        break

    if FLAGS.rollout_path:
      with open(FLAGS.rollout_path, 'wb') as f:
        pickle.dump(trajectories, f)
      logging.info('Saved trajectories to %s', FLAGS.rollout_path)

def main(argv):
  del argv
  tf.enable_resource_variables()
  tf.disable_eager_execution()

  learned_model = model.EncodeProcessDecode(output_size=PARAMS['size'], 
                                            latent_size=128, num_layers=2, message_passing_steps=15)

  if FLAGS.mode == 'train':
    learner(learned_model)
  elif FLAGS.mode == 'eval':
    run_eval(learned_model)

if __name__ == '__main__':
  app.run(main)