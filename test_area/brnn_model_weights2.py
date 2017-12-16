from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
import util
import reader
import subprocess
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, MultiRNNCell
from tensorflow.contrib.distributions import Normal
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

"""
    Global variables
"""
"""
model_type = "test"
data_path = "../data/"
save_path = "./saved_model"
global_prior_pi = 0.25
global_log_sigma1 = -1.0
global_log_sigma2 = -7.0
global_random_seed = 12
"""

def data_type():
    return tf.float32


def get_config():
    """Get model config."""
    if model_type == "small":
        config = SmallConfig()
    elif model_type == "medium":
        config = MediumConfig()
    elif model_type == "large":
        config = LargeConfig()
    elif model_type == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", model_type)

    config.prior_pi = global_prior_pi
    config.log_sigma1 = global_log_sigma1
    config.log_sigma2 = global_log_sigma2

    return config


def assigner(wd):
    """
    Can reassign weights to traiable variables
    INPUT: A dictionary with the sampled weights {name:value}
    OUTPUT: a list to be run in session such that all the tf.assigns give
            new values (assignments) to the trainable variables
    """
    assigns = []
    for t in tf.trainable_variables():
        if t.name in wd.keys():
            assigns.append(tf.assign(t, tf.convert_to_tensor(wd[t.name])))
    return assigns

def sample_network(names, variables, num_layers):
    """
    INPUT: 2 Lists, and number of layers in network
    OUTPUT: Dictionary mapping variable name (key) to
            variable values (numpy array of sampled weights)) 
    """
    LoT = list(zip(names, variables)) #List of Tuples
    out = {}
    
    #Embeddings
    mean = [entry for entry in LoT if "embedding_mean" in entry[0]][0]
    rho = [entry for entry in LoT if "embedding_rho" in entry[0]][0]
    #Sampling
    dim = mean[1].shape
    embedding = np.random.normal(mean[1].flatten(), np.absolute(rho[1].flatten())).reshape(dim)
    out[str(mean[0])] = embedding
    out[str(rho[0])] = rho[1]
    
    #Softmax 
    w_mean = [entry for entry in LoT if "softmax_w_mean" in entry[0]][0]
    b_mean = [entry for entry in LoT if "softmax_b_mean" in entry[0]][0]
    w_rho = [entry for entry in LoT if "softmax_w_rho" in entry[0]][0]
    b_rho = [entry for entry in LoT if "softmax_b_rho" in entry[0]][0]
    #Sampling
    dim = w_mean[1].shape
    dim2 = b_mean[1].shape
    softmax_w = np.random.normal(w_mean[1].flatten(), np.absolute(w_rho[1].flatten())).reshape(dim)
    softmax_b = np.random.normal(b_mean[1].flatten(), np.absolute(b_rho[1].flatten())).reshape(dim2)
    out[str(w_mean[0])] = softmax_w; out[str(w_rho[0])] = w_rho[1]
    out[str(b_mean[0])] = softmax_b; out[str(b_rho[0])] = b_rho[1]

    #Cell 0 
    w_mean = [entry for entry in LoT if "0_weights_mean" in entry[0]][0]
    b_mean = [entry for entry in LoT if "0_biases_mean" in entry[0]][0]
    w_rho = [entry for entry in LoT if "0_weights_rho" in entry[0]][0]
    b_rho = [entry for entry in LoT if "0_biases_rho" in entry[0]][0]
    #Sampling
    dim = w_mean[1].shape
    dim2 = b_mean[1].shape
    c0_w = np.random.normal(w_mean[1].flatten(), np.absolute(w_rho[1].flatten())).reshape(dim)
    c0_b = np.random.normal(b_mean[1].flatten(), np.absolute(b_rho[1].flatten())).reshape(dim2)
    out[str(w_mean[0])] = c0_w; out[str(w_rho[0])] = w_rho[1]
    out[str(b_mean[0])] = c0_b; out[str(b_rho[0])] = b_rho[1]

    
    if num_layers == 2:
        #Cell 1 
        w_mean = [entry for entry in LoT if "1_weights_mean" in entry[0]][0]
        b_mean = [entry for entry in LoT if "1_biases_mean" in entry[0]][0]
        w_rho = [entry for entry in LoT if "1_weights_rho" in entry[0]][0]
        b_rho = [entry for entry in LoT if "1_biases_rho" in entry[0]][0]    
        #Sampling
        dim = w_mean[1].shape
        dim2 = b_mean[1].shape
        c1_w = np.random.normal(w_mean[1].flatten(), np.absolute(w_rho[1].flatten())).reshape(dim)
        c1_b = np.random.normal(b_mean[1].flatten(), np.absolute(b_rho[1].flatten())).reshape(dim2)
        out[str(w_mean[0])] = c1_w; out[str(w_rho[0])] = w_rho[1]
        out[str(b_mean[0])] = c1_b; out[str(b_rho[0])] = b_rho[1]
    return out


def organize_weights(names, variables, num_layers):
    """
    Description: Tensorflow outputs trainable variables in any order. Here
                 we organize them into (mu, sigma) tuples for each weight block
    INPUT: 2 Lists, and number of layers in network
    OUTPUT: 7 tuples, where each tuple holds (mu, sigma) values of given weight block 
    """
    LoT = list(zip(names, variables)) #List of Tuples
    #out = {}
    
    #Embeddings
    mean = [entry for entry in LoT if "embedding_mean" in entry[0]][0]
    rho = [entry for entry in LoT if "embedding_rho" in entry[0]][0]
    embed = (mean[1],rho[1])

    
    #Softmax 
    w_mean = [entry for entry in LoT if "softmax_w_mean" in entry[0]][0]
    b_mean = [entry for entry in LoT if "softmax_b_mean" in entry[0]][0]
    w_rho = [entry for entry in LoT if "softmax_w_rho" in entry[0]][0]
    b_rho = [entry for entry in LoT if "softmax_b_rho" in entry[0]][0]
    smaxw = (w_mean[1], w_rho[1])
    smaxb = (b_mean[1], b_rho[1])


    #Cell 0 
    w_mean = [entry for entry in LoT if "0_weights_mean" in entry[0]][0]
    b_mean = [entry for entry in LoT if "0_biases_mean" in entry[0]][0]
    w_rho = [entry for entry in LoT if "0_weights_rho" in entry[0]][0]
    b_rho = [entry for entry in LoT if "0_biases_rho" in entry[0]][0]
    cell0w = (w_mean[1], w_rho[1])
    cell0b = (b_mean[1], b_rho[1])


    
    if num_layers == 2:
        #Cell 1 
        w_mean = [entry for entry in LoT if "1_weights_mean" in entry[0]][0]
        b_mean = [entry for entry in LoT if "1_biases_mean" in entry[0]][0]
        w_rho = [entry for entry in LoT if "1_weights_rho" in entry[0]][0]
        b_rho = [entry for entry in LoT if "1_biases_rho" in entry[0]][0]    
        cell1w = (w_mean[1], w_rho[1])
        cell1b = (b_mean[1], b_rho[1])
    return embed, smaxw, smaxb, cell0w, cell0b, cell1w, cell1b
    


def sample_weights(shape, mu, rho, seed):

    """
        Get a sample from the multivariate posterior
    """
    epsilon = Normal(0.0, 1.0).sample(shape, seed=seed)
    sigma = tf.nn.softplus(rho) + 1e-5
    output = mu + sigma * epsilon
    tf.assign(mu, output)
    print("EPSILON", epsilon)

    



def sample_posterior(shape, name, prior, is_training, is_hyper, seed):

    """
        Get a sample from the multivariate posterior
    """
    
    
    rho_max_init = math.log(math.exp(prior.sigma_mix / 2.0) - 1.0)
    rho_min_init = math.log(math.exp(prior.sigma_mix / 4.0) - 1.0)
    
    init = tf.random_uniform_initializer(rho_min_init, rho_max_init)
    
    with tf.variable_scope("BBB", reuse = not is_training):
        mu = tf.get_variable(name + "_mean", shape = shape, dtype=data_type())
    
    with tf.variable_scope("BBB", reuse = not is_training):
        rho = tf.get_variable(name + "_rho", shape = shape, dtype=data_type(), initializer=init)
        
    if is_training:
        epsilon = Normal(0.0, 1.0).sample(shape)
        sigma = tf.nn.softplus(rho) + 1e-5
        output = mu + sigma * epsilon  
    if is_hyper:
        epsilon = Normal(0.0, 1.0).sample(shape, seed=seed)
        sigma = tf.nn.softplus(rho) + 1e-5
        output = mu + sigma * epsilon
    else:
        output = mu
    
    if is_hyper:
        print(name, epsilon)
    
    if not is_training:
        return output
    
    tf.summary.histogram(name + '_rho_hist', rho)
    tf.summary.histogram(name + '_mu_hist', mu)
    tf.summary.histogram(name + '_sigma_hist', sigma)

    sample = output
    kl = get_kl_divergence(shape, tf.reshape(mu, [-1]), tf.reshape(sigma, [-1]), prior, sample)
    tf.add_to_collection('KL_layers', kl)

    return output


def get_kl_divergence(shape, mu, sigma, prior, sample):
    

    """
    Compute KL divergence between posterior and prior.
    log(q(theta)) - log(p(theta)) where
    p(theta) = pi*N(0,sigma1) + (1-pi)*N(0,sigma2)
    
    shape = shape of the sample we want to compute the KL of
    mu = the mu variable used when sampling
    sigma= the sigma variable used when sampling
    prior = the prior object with parameters
    sample = the sample from the posterior
    """
    
    #Flatten to a vector
    sample = tf.reshape(sample, [-1])
    
    #Get the log probability distribution of your sampled variable
    #So essentially get: q( theta | mu, sigma )
    posterior = Normal(mu, sigma)
    
    
    prior_1 = Normal(0.0, prior.sigma1)
    prior_2 = Normal(0.0, prior.sigma2)
    
    #get: sum( log[ q( theta | mu, sigma ) ] )
    q_theta = tf.reduce_sum(posterior.log_prob(sample))
    
    #get: sum( log[ p( theta ) ] ) for mixture prior
    mix1 = tf.reduce_sum(prior_1.log_prob(sample)) + tf.log(prior.pi_mix)
    mix2 = tf.reduce_sum(prior_2.log_prob(sample)) + tf.log(1.0 - prior.pi_mix)
    
    #Compute KL distance
    KL = q_theta - tf.reduce_logsumexp([mix1,mix2])
    
    return KL


class Prior(object):

    """
        For creating our fixed prior containing the desired 
        properties we want to use in the model.
        
        Setting pi = 1 will lead to a non mixture gaussian
        with mean zero and log var = log_sigma1
    """

    def __init__(self, pi, log_sigma1, log_sigma2):
        self.pi_mix = pi
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = tf.exp(log_sigma1)
        self.sigma2 = tf.exp(log_sigma2)
        sigma_one, sigma_two = math.exp(log_sigma1), math.exp(log_sigma2)
        self.sigma_mix = np.sqrt(pi * np.square(sigma_one) + (1.0 - pi) * np.square(sigma_two))


class BayesianLSTMCell(BasicLSTMCell):
    def __init__(self, num_units, prior, is_training, is_hyper, seed, name=None, **kwargs):
        
        super(BayesianLSTMCell, self).__init__(num_units, **kwargs)

        self.w = None
        self.b = None
        self.prior = prior
        self.n = name
        self.is_training = is_training
        self.is_hyper = is_hyper
        self.seed = seed
        self.num_units = num_units
        

    def call(self, inputs, state):
        if self.w is None:

            size = inputs.get_shape()[-1].value
            
            self.w = sample_posterior((size + self.num_units, 4 * self.num_units),
                                          name=self.n + "_weights",
                                          prior=self.prior,
                                          is_training=self.is_training,
                                          is_hyper=self.is_hyper,
                                          seed=self.seed)
            
            neuron_w = self.w[0]
            tf.summary.histogram(self.n + '_neuronW_hist', neuron_w)

            self.b = sample_posterior((4 * self.num_units, 1),
                                           name=self.n + "_biases",
                                           prior=self.prior,
                                           is_training=self.is_training,
                                           is_hyper=self.is_hyper,
                                           seed=self.seed)

        cell, hidden = state
        
        concat_inputs_hidden = tf.concat([inputs, hidden],1)

        concat_inputs_hidden = tf.nn.bias_add(tf.matmul(concat_inputs_hidden, self.w), tf.squeeze(self.b))
        
        i, j, f, o = tf.split(value=concat_inputs_hidden, num_or_size_splits=4, axis=1)

        new_cell = (cell * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
        
        new_hidden = self._activation(new_cell) * tf.sigmoid(o)

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


class PTBModel(object):
    def __init__(self, is_training, is_hyper,seed, config, input_):
        self._is_training = is_training
        self._is_hyper = is_hyper
        self._input = input_
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        self.size = size = config.hidden_size
        self.num_layers = config.num_layers
        vocab_size = config.vocab_size
        
        # Construct prior
        prior = Prior(config.prior_pi, config.log_sigma1, config.log_sigma2)
        
        # Fetch embeddings
        with tf.device("/cpu:0"):
            embedding = sample_posterior([vocab_size, size], "embedding", prior, is_training, is_hyper, seed=seed)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
     
            
            # Declare labels
            self.labels_id = input_.targets
            self.labels = tf.nn.embedding_lookup(embedding, input_.targets)
       
        
            
        # Build the BBB LSTM cells
        cells = []
        for i in range(config.num_layers):
            cells.append(BayesianLSTMCell(size, prior, is_training,
                                          is_hyper=is_hyper,
                                      name = "bbb_lstm_{}".format(i),
                                      forget_bias=0.0, seed=seed))

        cell = MultiRNNCell(cells, state_is_tuple=True)
        self._initial_state = cell.zero_state(config.batch_size, data_type())
     
        ##### PROBLEM: how do we give a name to the initial state operation  ##########
        #####          so that we can restore it and do inference in restore.py? ######
        
        #state = tf.add(self._initial_state, 0, name="INITIAL_STATE")
        #state = tf.tuple((self._initial_state[0],self._initial_state[1]), name="INITIAL_STATE")
        
        #c = tf.zeros([self.num_layers, self.batch_size, self.size], tf.float32, name="initial_state_c")
        #h = tf.zeros([self.num_layers, self.batch_size, self.size], tf.float32, name="initial_state_h")
        #self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        #self._initial_state = tf.tuple( (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),), name= "INITIAL_STATE0")
        
        state = self._initial_state
        
        """
        # Type inside tuple
        print("initial_state type: ", type(self._initial_state))
        print("initial_state.c type: ", type(self._initial_state[0]))
        print("initial_state.h type: ", type(self._initial_state[1]))
        
        # Type inside tuple of tuples
        print("initial_state.c1 type: ", type(self._initial_state[0][0]))
        print("initial_state.h2 type: ", type(self._initial_state[1][1]))
        
        # Shape
        print("initial_state.c: ",      self._initial_state[0][0])
        print("initial_state.h: ",      self._initial_state[1][1])
        print("initial_state.c1 shape: ", self._initial_state[0][0].shape)
        print("initial_state.h2 shape: ", self._initial_state[1][1].shape)
        """
        
        
        # Forward pass for the truncated mini-batch
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

            
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

        # Softmax weights
        softmax_w = sample_posterior((size, vocab_size), "softmax_w", prior, is_training, is_hyper, seed=seed)
        softmax_b = sample_posterior((vocab_size, 1), "softmax_b", prior, is_training, is_hyper, seed=seed)
        
    
        if is_hyper:
            # Pull original weights into dict
            wd1, names1, values1 = extract_weights() 
            
            weights = [embed, smaxw, smaxb, cell0w, cell0b, cell1w, cell1b] = organize_weights(names1, values1, 2)

            # Sample new weights based on original
            for entry in weights:
                _ = sample_weights(shape=tf.shape(entry[0]), mu=entry[0], rho=entry[1], seed=seed)
            
            
            
            # Replace original with sampled weights into Graph
            #new_w = assigner(wd2)
            
            
            # Verify graph's trainable vars have new assigned (sampled) values
            #wd, _, _ = extract_weights() 
            #print_weights(wd, "NEW ASSIGNMENTS")
            
        # Predictions 
        logits = tf.nn.xw_plus_b(output, softmax_w, tf.squeeze(softmax_b))
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size], name="LOGITS")
        self.logits = logits
        
        
        # Calculate Loss
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=False)

        # Update the cost
        # Remember to divide by batch size
        self._cost = tf.divide(tf.reduce_sum(loss), tf.cast(self.batch_size, tf.float32), name="COST")
        self._kl_loss = 0.

        self._final_state = state
        state = tf.add(state, 0, name="FINAL_STATE")
        

        if not is_training:
            return

        #Compute KL divergence
        #B = number of batches aka the epoch size
        #C = number of truncated sequences in a batch aka batch_size variable
        B = self._input.epoch_size
        C = self.batch_size
        
        kl_loss = tf.add_n(tf.get_collection("KL_layers"), "kl_divergence")
        
        kl_factor = 1.0/(B*C)
        self._kl_loss = kl_factor * kl_loss
        
        self._total_loss = self._cost + self._kl_loss

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._total_loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)



    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost,
               util.with_prefix(self._name, "kl_div"): self._kl_loss,
               util.with_prefix(self._name, "logits"): self.logits,
               util.with_prefix(self._name, "labels_id"): self.labels_id,
               util.with_prefix(self._name, "initial_state"): self._initial_state}
        
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
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
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        self._kl_loss = tf.get_collection_ref(util.with_prefix(self._name, "kl_div"))[0]
        self.logits = tf.get_collection_ref(util.with_prefix(self._name, "logits"))[0]
        self.labels_id = tf.get_collection_ref(util.with_prefix(self._name, "labels_id"))[0]
        self._initial_state = tf.get_collection_ref(util.with_prefix(self._name, "initial_state"))
        num_replicas = 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        """
        c = tf.zeros([self.num_layers, self.batch_size, self.size],
                 tf.float32)
        h = tf.zeros([self.num_layers, self.batch_size, self.size],
                 tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        """
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

    @property
    def kl_loss(self):
        return self._kl_loss if self._is_training else tf.constant(0.)


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


class MediumConfig(object):
    """
    Medium config.
    Slightly modified according to email.
    """
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 20
    max_max_epoch = 70
    keep_prob = 1.0
    lr_decay = 0.9
    batch_size = 20
    vocab_size = 10000


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


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 2
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def extract_weights():
    """ Does not require session input"""
    names = [v.name for v in tf.trainable_variables()]
    values = [v for v in tf.trainable_variables()]
    weights_dict = dict(zip(names, values))
    return weights_dict, names, values

def extract_weights2(session):
    """ To be run in session"""
    names = [v.name for v in tf.trainable_variables()]
    values = session.run(names)
    weights_dict = dict(zip(names, values))
    return weights_dict, names, values


def print_weights(wd, name):
    for k, v in wd.items():
        print("\n\n", name, " VARIABLE: ", k)
        print("SHAPE: ", v.shape)
        print(v)
        
def run_epoch(session, model, data, id_to_word, is_hyper, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    
    state = session.run(model.initial_state)  # just feed it zeros in a structure of final state
    
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
        fetches["kl_divergence"] = model.kl_loss

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
        
        #Print the first 100 predictions vs targets
        
        if model._name == "Test" and step <= 100:
            word_logit = session.run(model.logits)
            single_word = word_logit[0,0,:]
            print("Most probable word is: ",id_to_word[int(np.argmax(single_word))])
            targets = session.run(model.labels_id)
            single_target = targets[0,0]
            print("Target is : ", id_to_word[int(single_target)]) #data[step])
            

        if verbose and (step % (model.input.epoch_size // 10) == 10 or step == 0):
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

            if model._is_training:
                print("KL is {}".format(vals["kl_divergence"]))
                
    return np.exp(costs / iters) 

#def change_random_seed(seed):
#    global prng
#    prng = np.random.RandomState(seed)
#    tf.set_random_seed(seed)


def main( model_select="test",
         dat_path = "../data",
         sav_path = "./tensorboard_test/",
         mixing_pi = 0.25,
         prior_log_sigma1 = -1.0,
         prior_log_sigma2 = -7.0):
    
    global model_type
    global data_path
    global save_path
    global global_prior_pi
    global global_log_sigma1
    global global_log_sigma2
    #    global global_random_seed
    
    model_type = model_select
    data_path = dat_path
    save_path = sav_path
    global_prior_pi = mixing_pi
    global_log_sigma1 = prior_log_sigma1
    global_log_sigma2 = prior_log_sigma2
    #    global_random_seed = set_random_seed
    
    #    change_random_seed(global_random_seed)
    raw_data = reader.ptb_raw_data(data_path)
    train_data, valid_data, test_data, _, id_to_word = raw_data
    
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    #eval_config.num_steps = 1
    
    
    
    
    
    subprocess.Popen(["tensorboard","--logdir=tensorboard", "--port=6007"])
    
############         ORIGINAL MODEL: TRAIN, VAL, TEST       #############################
    
    
    ##############      BUILD MULTIPLE GRAPHS       #################33
    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
    
        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, is_hyper=False, config=config, input_=train_input, seed=1)
            tf.summary.scalar("Training_Loss", m.cost)
            tf.summary.scalar("Learning_Rate", m.lr)
    
        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, is_hyper=False, config=config, input_=valid_input, seed=1)
            tf.summary.scalar("Validation_Loss", mvalid.cost)
    
        with tf.name_scope("Test"):
            test_input = PTBInput(
                config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, is_hyper=False, config=eval_config,
                                 input_=test_input, seed=1)


        models = {"Train": m, "Valid": mvalid, "Test": mtest}        
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        soft_placement = False
        
    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
    
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data, id_to_word, is_hyper=False, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
    
                valid_perplexity = run_epoch(session, mvalid, valid_data, id_to_word,is_hyper=False)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
    
            test_perplexity = run_epoch(session, mtest, test_data, id_to_word,is_hyper=False)
            print("Test Perplexity: %.3f" % test_perplexity)
            wd, _, _ = extract_weights()
            
            if save_path:
                print("Saving model to %s." % save_path)
                sv.saver.save(session, save_path + "model1", global_step=sv.global_step)     
    


"""
#################       HYPERNETS       #########################################


    ##########      HYPER0          ######################
    
    # Start Graph
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        # Run Session        
        with tf.Session() as session:
            
            # Restore Graph 
            saver = tf.train.Saver 
            latest_checkpoint = tf.train.latest_checkpoint("./tensorboard_test/model1")

        
            # Restore previously trained variables from disk
            saver.restore(session, latest_checkpoint)
            # Build Graph
            with tf.name_scope("Test"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mhyper0 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                     input_=test_input, seed=2)             
            
            
    
           
                
                hyper_perplexity = run_epoch(session, mhyper0, test_data, id_to_word,is_hyper=True)
                print("Hyper Perplexity0: %.3f" % hyper_perplexity)
                wd, _, _ = extract_weights2(session)
                print_weights(wd, name="Hyper0: ".format(i))
            



    ##########      HYPER01         ######################
    
    # Start Graph
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        # Build Graph
        with tf.name_scope("Hyper1"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper1 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=2)
                
        models = {"Hyper1": mhyper1}        
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        
        # Run Session        (Evaluate Graph)         
    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            hyper_perplexity = run_epoch(session, mhyper1, test_data, id_to_word,is_hyper=True)
            print("Hyper Perplexity1: %.3f" % hyper_perplexity)
            wd, _, _ = extract_weights2(session)
            print_weights(wd, name="Hyper1: ".format(i))


    ########### SAVE MODEL      ###########################



    ##########      HYPER1          ######################
    
    # Start Graph
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        # Build Graph
        with tf.name_scope("Hyper1"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper1 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=2)
                
       # Run Session        (Evaluate Graph)  
        with sv as session:   
            hyper_perplexity = run_epoch(session, mhyper1, test_data, id_to_word,is_hyper=True)
            print("Hyper Perplexity1: %.3f" % hyper_perplexity)
            wd, _, _ = extract_weights2(session)
            print_weights(wd, name="Hyper1: ".format(i))
        

    ##########      HYPER2          ######################
    
    # Start Graph
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        # Build Graph
        with tf.name_scope("Hyper2"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper2 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=2)
                
       # Run Session        (Evaluate Graph)  
        with sv as session:   
            hyper_perplexity = run_epoch(session, mhyper2, test_data, id_to_word,is_hyper=True)
            print("Hyper2 Perplexity: %.3f" % hyper_perplexity)
            wd, _, _ = extract_weights2(session)
            print_weights(wd, name="Hyper2: ".format(i))
        
"""
    

                     
            
"""                
                
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
                
        with tf.name_scope("Hyper1"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper1 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=3)
                
                
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
                
        with tf.name_scope("Hyper2"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper2 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=4)
                
        with tf.name_scope("Hyper3"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper3 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=5)
                
        with tf.name_scope("Hyper4"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper4 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=6)
                
        with tf.name_scope("Hyper5"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper5 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=7)
                
        with tf.name_scope("Hyper6"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mhyper6 = PTBModel(is_training=False, is_hyper=True, config=eval_config,
                                 input_=test_input, seed=8)
                
                
        
        models = {"Train": m, "Valid": mvalid, "Test": mtest, "Hyper0": mhyper0,
                  "Hyper1": mhyper1,"Hyper2": mhyper2,"Hyper3": mhyper3,"Hyper4": mhyper4,
                  "Hyper5": mhyper5, "Hyper6": mhyper6}
        hyper_models = [mhyper0, mhyper1,mhyper2,mhyper3, mhyper4, mhyper5, mhyper6]
        
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        soft_placement = False
        
    
        #hyper = {"Hyper": mhyper}
        #mhyper.export_ops("Hyper")
        #metagraph_hyper = tf.train.export_meta_graph()
        
        
    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
    
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data, id_to_word, is_hyper=False, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
    
                valid_perplexity = run_epoch(session, mvalid, valid_data, id_to_word,is_hyper=False)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
    
            test_perplexity = run_epoch(session, mtest, test_data, id_to_word,is_hyper=False)
            print("Test Perplexity: %.3f" % test_perplexity)
            wd, _, _ = extract_weights()
            print_weights(wd, name="Test: ".format(i))
            
            ensemble_size = len(hyper_models)
            hyper_results = []
            for i in range(ensemble_size):
                hyper_results.append(run_epoch(session, hyper_models[i], test_data, id_to_word, is_hyper=True))
                print("Hyper Perplexity{}: ".format(i),  hyper_results[i])
                wd, name, val = extract_weights()
                for v in val:
                    print(session.run(v))
                #print_weights(val.eval(), name="Hyper{}: ".format(i))
            #hyper_perplexity = run_epoch(session, mhyper, test_data, id_to_word)
            
    
            
  
        
        # Extract and print trained weights
        #weights = session.run(tf.trainable_variables())
        #names = [v.name for v in tf.trainable_variables()]
        #values = session.run(names)
        #weights_dict = dict(zip(names, values))
        #for k, v in zip(names, values):
        #    print("Variable: ", k)
        #    print("Shape: ", v.shape)
         #   print("\nTRAINED VARIABLE: \n", v)
        
"""        
"""        
        # Sample new weights based on trained weights,
        # and assign placeholders with new sampled weights
        wdict1 = sample_network(names, values, 2)
        for k, v in wdict1.items():
            print("Sampled Variable: ", k)
            print("Shape: ", v.shape)
            print("\nSAMPLED VARIABLE: \n", v)
            

        #networks = sample_network(names, values, config.num_layers) for i in range(10)]
        print("\nLet's stop here\n")
        assigns = assigner(wdict1)
        session.run(assigns)
        
        # Check if new assignments were given
        names = [v.name for v in tf.trainable_variables()]
        values = session.run(names)
        #weights_dict = dict(zip(names, values))
        for k, v in zip(names, values):
            print("Assigned Variable: ", k)
            print("Shape: ", v.shape)
            print("\nASSIGNED VARIABLE: \n", v)
 
"""       
        

            
            
"""           
with tf.Graph().as_default():
    #reset()
    save("sampled_network1", wdict1, save_path)
    print_tensors_in_checkpoint_file(
           file_name='tensorboard/sampled_network1', #.ckpt.data-00000-of-00001', 
            tensor_name='', 
            all_tensors=True)
    print("Finished print_tensors_in_checkpoint()")
            
#with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph_hyper) # , import_scope="Test")
    print("Finished import_meta_graph")
    mhyper.import_ops()
    print("Finished myper.import_ops()")
    restore(save_path, "sampled_network1",wdict1)
    print("Finished restore()")
    
    sv = tf.train.Supervisor(logdir=save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
        hypernet_perplexity = run_epoch(session, mhyper)
        print("Hypernet Test Perplexity: %.3f" % hypernet_perplexity)
        
"""           



if __name__ == '__main__':
    main()