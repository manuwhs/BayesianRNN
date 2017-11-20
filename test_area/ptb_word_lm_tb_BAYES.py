
"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import os

import sys

import reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "test",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 0,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def sample_random_normal(name, mean, std, shape):
    
    with tf.variable_scope("sample_random_normal"):
    
        #Inverse softplus (positive std)
        standard_dev = tf.log(tf.exp(std) - 1.0) * tf.ones(shape)
        
        mean = tf.get_variable(name + "_mean", initializer=mean, dtype=tf.float32)
        standard_deviation = tf.get_variable(name + "std", initializer=std, dtype=tf.float32)
        #Revert back to std
        standard_deviation = tf.nn.softplus(standard_deviation)
    
        #Sample standard normal
        epsilon = tf.random_normal(mean=0, stddev=1, name="epsilon", shape=shape, dtype=tf.float32)
      
        random_var = mean + standard_deviation*epsilon
    
    return random_var

class BayesianLSTMCell(BasicLSTMCell):
    def __init__(self, num_units, Weights, Biases **kwargs):
        
        #self.mean = mean
        #self.std = std
        self.num_units = num_units
        self.Weights = Weights
        self.Biases = Biases
        
        #From BasicLSTMCell
        super(BayesianLSTMCell, self).__init__(num_units, **kwargs)
    
    
    """
        A note on the shape for the sampling of weights and biases:
        
        The number of weights we want to sample is determined by the hidden state
        from the previous cell, the input data, and of course the number of gates
        in a single cell.
        
        Assume that num_units = 10:
        There's 4 gates in an LSTM cell. Hence when we say we want num_units = 10
        then the LSTM cell actually consists of 4*num_units = 40 hidden units.
        
        The hidden state from the previous gate will have the shape h=[num_units]=[10]
        The input data is embedded and in the original model embedding_size = num_units
        This can of course be changed if wanted. So input data will have the shape
        inputs=[embedding_size]=[num_units]=[10]
        
        Input data and the hidden state from the previous cell is concatenated before passed
        to any gate. Thus the input to each gate will be x = [embedding_size + num_units] = [20]
        that is, a 20 long vector.
        
        So the total amount of weights needed will be 4*num_units*(embedding_size+num_units) = 160
        The total amount of biases is just the length of the input vector x to the gates, so the
        total number of biases should be embedding_size + num_units = 20
    """
    
    
    #Class call function
    def __call__(self, inputs, state):
        with tf.variable_scope("BayesLSTMCell"):
            
            #State is a tuple with the cell and hidden state vectors from
            #the previous BayesianLSTMCell
            cell, hidden = state
            
            #Vector concatenation of previous hidden state and embedded inputs
            concat_inputs_hidden = tf.concat([inputs, hidden], 1)
            
            
            """
                gate_inputs is basically the calculation Wx + b of ALL gates.
                Take e.g. num_units = 2. Thus total number of hidden_units = 8.
                The input vector x in this case is a 2 long vector, as is the hidden
                state vector. So dimensions are W = 4x8, x = 4 and b = 8.
                Then we can do Wx + b and get an 8 long vector which can be passed
                through the 4 gates and their respective activation functions.
            """
            gate_inputs =  tf.nn.bias_add(tf.matmul(concat_inputs_hidden, Weights), Biases)

            #Split data up for the 4 gates
            #i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(gate_inputs, axis = 1, num_or_size_splits = 4)

            """
                Calculate new cell and new hidden states. Calculations are as in Zaremba et al 2015:
                
                new_cell = cell*\sigma(f + bias) + \sigma(i)*\sigma(j)
                new_hidden = \tanh(new_cell)*\sigma(o)
                
                See the LSTM graph here: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
                
            """
            new_cell = (cell * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i)*self._activation(j))
            new_hidden = self._activation(new_cell) * tf.sigmoid(o)
            
            #Create tuple of the new state
            new_state = LSTMStateTuple(new_cell, new_hidden)

            return new_hidden, new_state



class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)
    print("Batch Size: ", batch_size)
    print("Input Data: ", self.input_data.size)
    print("Target Data: ", self.targets.size)




class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      print("Inputs (output of embedding_lookup): ", inputs.size)

    # We comment 2 lines below because we don't implement DropOut; we implement Bayes by Backprop thorugh Time instead  
    #if is_training and config.keep_prob < 1:
    #  inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph_lstm(inputs, config, is_training)
    print("Output: ", output.size)
    print("State: ", state.size)

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    print("softmax_w: ", softmax_w.size)
    print("softmax_b: ", softmax_b.size)
    print("logits: ", logits.shape)
     # Reshape logits to be a 3-D tensor for sequence loss
     # The logits correspond to the prediction across all classes at each timestep.
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
    print("logits reshaped: ", logits.shape)

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets, # this should be of shape: [batch_size, num_steps]
        # these ones are the weigths to the predictions: it's weighting each prediction in the sequence equally
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        # The combination of False, True below results in a loss of shape: [num_steps] (since it averages over batch)
        # This means that loss vector has elements that coresspond to the loss at each time step
        # These are later summed up with tf.reduce_sum(loss) below into one single scalar loss value  
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss) # this should now be a scalar. NOTE: if we skip this step and pass a loss
                                     # vector shape [num_steps] to tf.gradients() below, then we can obtain the
                                     # likelihood loss gradient (wrt to the weights) at each time step. I.e., we
                                     # will obtain 35 (num_steps) gradients
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables() # This convenience function calls all variables with trainable=True into a list
    grads, _ = tf.clip_by_global_norm( # All of our clipped gradients are now stored in grads as a list in order of tvars
                                      tf.gradients(self._cost, tvars), # This will construct symbolic derivatives:
                                                                       # dc_dw1, dc_dw2, ... dc_dwN (N = num_weights, num_tvars)
                                      config.max_grad_norm) # avoiding exploding gradients by clipping them

    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients( # Returns an an Operation that applies the specified gradients
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


  #Sample weights
  def sample_weights(self):
      with tf.variable_scope("CellWeights"):
          self.Weights = sample_random_normal("WeightMatrix",
                                              self.mean,
                                              self.std,
                                              shape = [2*size,4*size]) #[2*self.num_units,4*self.num_units])
          return self.Weights
  
  #Sample biases
  def sample_biases(self):
      with tf.variable_scope("CellBiases"):
          self.Biases = sample_random_normal("BiasVector",
                                             self.mean,
                                             self.std,
                                             shape = [4*size]) #[4*self.num_units])
          return self.Biases

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.

    Weights = self.sample_weights()
    Biases  = self.sample_biases()

    cell = BayesianLSTMCell(config.hidden_size, Weights, Biases) #config, is_training)
    

    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)], state_is_tuple=True)


    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    
    output = tf.concat(outputs,1) 
    output = tf.reshape(output, [-1, config.hidden_size])
    
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = CUDNN


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 5
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 10
  vocab_size = 10000
  rnn_mode = BLOCK

# Epoch size is 2323
def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    # Prints when remainder of step/232 == 10
    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config

def main(_):
    
  #Manually set flags here
  flags.FLAGS.data_path = "../../data/"
  flags.FLAGS.save_path = "tensorboard/"
        
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training_Loss", m.cost)
      tf.summary.scalar("Learning_Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation_Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    
    import subprocess
    subprocess.Popen(["tensorboard","--logdir=tensorboard"])
    
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
