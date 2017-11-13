#### Import ####
import subprocess
import reader
import tensorflow as tf

#For Bayesian LSTM cell implementation
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, MultiRNNCell

#### Functions ####

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


#### Borrowed from PTB model ####

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    #self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)
    

    
class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 2
  num_steps = 20
  hidden_size = 20
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  #rnn_mode = BLOCK


#### End of borrowed from PTB model ####
    

"""
    Bayesian LSTM Cell framework which inherits from tensorflows BasicLSTMCell
    
        Input parameters:
            On initialization:
                -mean = the mean to be used when sampling
                -std = the standard deviation to be used when sampling
                -num_units = the number of hidden units in each gate
                -**kwargs is the remaining arguments inherited from BasicLSTMCell
            
        On call:
                -inputs = input data i.e. embedded words
                -state = A state tuple (c,h) containing the cell and hidden state
                from the previous LSTM cell
        
        Output parameters:
            -On initialization: 
                Creates a BayesianLSTMCell class with inherited properties from BasicLSTMCell
                and the functions given in the class
            
            -On call:
                Outputs the current state tuple (c,h) containing the current cell and hidden
                state of the LSTMCell
    
"""
class BayesianLSTMCell(BasicLSTMCell):
    def __init__(self, mean, std, num_units, **kwargs):
        
        self.mean = mean
        self.std = std
        self.num_units = num_units
        self.Weights = None
        self.Biases = None
        
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
    
    #Sample weights
    def get_weights(self):
        with tf.variable_scope("CellWeights"):
            self.Weights = sample_random_normal("WeightMatrix",
                                                self.mean,
                                                self.std,
                                                shape = [2*self.num_units,4*self.num_units])
            return self.Weights
    
    #Sample biases
    def get_biases(self):
        with tf.variable_scope("CellBiases"):
            self.Biases = sample_random_normal("BiasVector",
                                               self.mean,
                                               self.std,
                                               shape = [4*self.num_units])
            return self.Biases
    
    #Class call function
    def __call__(self, inputs, state):
        with tf.variable_scope("BayesLSTMCell"):
            
            #State is a tuple with the cell and hidden state vectors from
            #the previous BayesianLSTMCell
            cell, hidden = state
            
            #Vector concatenation of previous hidden state and embedded inputs
            concat_inputs_hidden = tf.concat([inputs, hidden], 1)
            
            #Sample weights and biases
            Weights = self.get_weights()
            Biases = self.get_biases()
            
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


#Initiate Tensorboard

subprocess.Popen(["tensorboard","--logdir=tensorboard"])



#### Process data ####
##
## Currently we're only interested in running a minimal sample
## To verify graphs/sessions and such
##
####

tf.reset_default_graph()

### Set initial parameters

#Current data path
path = "../data/"

### Load data
raw_data = reader.ptb_raw_data(path)
train_data, valid_data, test_data, _ = raw_data

config = TestConfig()

train_input = PTBInput(config=config, data=train_data, name="TrainInput")

#collect input data and targets by calling the ptb_producer function
#input_data, targets = reader.ptb_producer(train_data, batch_size, num_steps, name = "TrainInput")

#Embed the data: Embedding format is [vocab_size, hidden_size] = [20, 2]
embedding = tf.get_variable("embedding", [config.vocab_size, config.hidden_size], dtype=tf.float32)
#inputs = [20, 1, 2] tensor containing 20 words for one timestep with 2 hidden units
input = tf.nn.embedding_lookup(embedding, train_input.input_data)


#### Build the graph ####



#Initialize a single instance of a Bayesian LSTM cell
cell = BayesianLSTMCell(mean=0.0, std=1.0, num_units=config.hidden_size, state_is_tuple=True)
#Create a 2 layer BRNN using the MultiRNNCell wrapper
cell = MultiRNNCell([cell for _ in range(config.num_layers)], state_is_tuple=True)

#Set initial state of the BRNN to zeros (i.e. at time zero we simply initialize the state to zero.)
with tf.variable_scope("InitialState"):
    state = cell.zero_state(batch_size=20, dtype=tf.float32)

#Unroll the BRNN for as many time steps as defined in config.num_steps
outputs = []
with tf.variable_scope("BRNN"):
    for time_step in range(config.num_steps):
        #We reuse the variable name for all unrolls
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        #Getting the output and the state for each time step
        (cell_output, state) = cell(inputs=input[:,time_step,:], state=state)
        #Collect outputs from cells in outputs array
        outputs.append(cell_output)
    #Reshaping. Not sure why
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

#Run a session on the graph

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #Create event file for tensorboard
    summary_write = tf.summary.FileWriter(("tensorboard"),sess.graph)
    
    #Feed data to a placeholder
    #feed_dict = {}
    #Run desired stuff
    #Y_out = sess.run(Y, feed_dict = feed_dict)
    
