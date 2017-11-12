#### Import ####

import subprocess
import reader
import tensorflow as tf

#For Bayesian LSTM cell implementation
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, MultiRNNCell

### My implementation ###
"""
lstm_cell = tf.contrib.rnn.BasicLSTMCell(650)
print(lstm_cell.state_size)
print(lstm_cell.output_size)
initial_state =  tf.zeros([20, 650]),  tf.zeros([20, 650])
print(len(initial_state))
"""
### Try 2: Zaremba Implementation: Adapted ###

#### ____ 1.  Architecture Dimensions ____ ###
# You can use the parameters of line below or just use: config=MediumConfig() 
num_steps=35; hidden_units=650; batch_size=20; layers=2; vocab_size = 10000;

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)
    
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
        print(random_var)
    return random_var

"""
    Bayesian LSTM Cell framework which inherits from tensorflows BasicLSTMCell
    
"""
class BayesianLSTMCell(BasicLSTMCell):
    def __init__(self,
                 mean, #mean for sampling weights
                 std, #standard deviation for sampling weights
                 config, 
                 embedding_size,
                 # --- below is inherited from BasicLSTMCell --- #
                 num_units, #The number of hidden units in the cell (4*num_units)
                 forget_bias = 1.0, #bias added to forget gates
                 input_size = None, #Deprecated and unused
                 state_is_tuple = True, #If True, accepted and returned states are 2-tuples of the c_state and h_state
                 activation = tf.tanh):
        
        self.mean = mean
        self.std = std
        self.Weights = None
        self.Biases = None
        
        self.config = config
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        #self.max_grad_norm = config.max_grad_norm
        #self.learning_rate = config.learning_rate
        #self.learning_rate_decay = config.learning_rate_decay
        #self.init_scale = config.init_scale
        #self.summary_frequency = config.summary_frequency
        #self.is_training = is_training
        
        self.embedding_size = embedding_size
        self.forget_bias = forget_bias #NOTE: for some reason they weren't being passed through, so did it explicitly
        self.activation = activation
        
        #From BasicLSTMCell
        #super().__init__()
        super(BayesianLSTMCell, self).__init__(num_units, forget_bias, input_size, state_is_tuple, activation)
    
    #Sample weights: dim = [self.embedding_size + self.hidden_size, 4 * self.hidden_size]
    def get_weights(self):
        with tf.variable_scope("CellWeights"):
            self.Weights = sample_random_normal("WeightMatrix", self.mean, self.std, 
                                                shape = [self.embedding_size + self.hidden_size, 4 * self.hidden_size]) #change shape to variables!
            print("\nWeights:\t", self.Weights)
            return self.Weights
    
    #Sample biases
    def get_biases(self):
        with tf.variable_scope("CellBiases"):
            self.Biases = sample_random_normal("BiasVector", self.mean, self.std, 
                                               shape = [4 * self.hidden_size]) #change shape to variable!
            print("\nBiases:\t", self.Biases)
            return self.Biases
    
    #Object call function
    def __call__(self, inputs, state):
        with tf.variable_scope("BayesLSTMCell"):  # "BasicLSTMCell"
            
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
            
            
            all_inputs = tf.concat([inputs, h], 1)
            print("\nInputs:\t", inputs); print("\nall_inputs:\t", all_inputs)
            concat = tf.nn.bias_add(tf.matmul(all_inputs, self.get_weights()), self.get_biases()) #self.Biases)
            print("\nconcat:\t", concat)
            
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
            print("\nI Gate:\t", i);print("\nJ Gate:\t", j);print("\nF Gate:\t", f);print("\nO Gate:\t", o);
            
            #Calculate new cell and hidden states. Calculations are as in Zaremba et al 2015
            new_c = (c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * self.activation(j))
            new_h = self.activation(new_c) * tf.sigmoid(o)
            
            
            #Create tuple of the new state
            new_state = LSTMStateTuple(new_c, new_h)

            return new_h, new_state
        
        
#### ____ 2. Loading 1 batch of data ___ ###
#input_data is [20, 35] tensor of the data prepared by reader

#Current data path
path = "../data/"

### Load data
raw_data = reader.ptb_raw_data(path)
train_data, valid_data, test_data, _ = raw_data

config=MediumConfig()
train_input = PTBInput(config=config, data=train_data, name="TrainInput")
print("Train_input contains a pair of Tensors (.input_data, .targets), each shaped [batch_size, num_steps]. The second element\
    of the tuple is the same data time-shifted to the right by one.\n\n ","train_input = ", train_input.input_data)

input_data = train_input.input_data



#tf.reset_default_graph()

#### ____ 3. Setting up LSTM RNN ____ ###
#cell = tf.contrib.rnn.BasicLSTMCell(650, forget_bias=0.0, state_is_tuple=True)
#cellm = tf.contrib.rnn.MultiRNNCell([cell for _ in range(layers)], state_is_tuple=True)
cell = BayesianLSTMCell(mean=0.0, std=1.0, config=config, embedding_size=config.hidden_size, num_units=config.hidden_size)


initial_state = cell.zero_state(20, tf.float32) #cellm.zero_state(20, tf.float32)
state = initial_state
print("\nState:\t", state)

embedding = tf.get_variable("embedding", [vocab_size, hidden_units], tf.float32)
inputs = tf.nn.embedding_lookup(embedding, input_data); print("\nInputs:\t",inputs); 
#inputs dimensions: [vocab_size, hidden_units] : this comes from word embeddings

#(cell_output, state1) = cell(inputs[:, 0, :], state)
#print("\nCell Output:\t", cell_output);   print("\nState:\t", state1)



outputs = []; 
for time_step in range(num_steps):
    (cell_output, state) = cell(inputs[:, time_step, :], state)
    outputs.append(cell_output)
output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_units])
print("\nOutput:\t", output);   print("\nState:\t", state)