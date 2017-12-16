import tensorflow as tf
import numpy as np
import reader
import math
import time 
import brnn_model_weights2 as bm
from tensorflow.python.framework import ops
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, MultiRNNCell


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

def extract_weights(session):
    names = [v.name for v in tf.trainable_variables()]
    values = session.run(names)
    weights_dict = dict(zip(names, values))
    return weights_dict, names, values

def print_weights(wd, name):
    for k, v in wd.items():
        print("\n\n", name, " VARIABLE: ", k)
        print("SHAPE: ", v.shape)
        print(v)
        
def assigner(wd):
    assigns = []
    for t in tf.trainable_variables():
        if t.name in wd.keys():
            assigns.append(tf.assign(t, tf.convert_to_tensor(wd[t.name], dtype=tf.float32)))
    return assigns

def create_vars(wd):
    for name, value in wd.items():
        tf.constant(name=name, value=value)


#####################################################



# Restore Graph 
latest_checkpoint = tf.train.latest_checkpoint("./tensorboard_test/")
saver = tf.train.import_meta_graph(latest_checkpoint + ".meta") #("./tensorboard/model1-23239.meta")


        
session = tf.Session()


# Restore previously trained variables from disk
saver.restore(session, latest_checkpoint) #"./tensorboard/model1-23239")


# Pull original weights into dict
wd1, names1, values1 = extract_weights(session) 
print_weights(wd1, "TRAINED")


# Sample new weights based on original
wd2 = sample_network(names1, values1, 2)
print_weights(wd2, "SAMPLED")


# Replace original with sampled weights into Graph
new_w = assigner(wd2)
session.run(new_w)


# Verify graph's trainable vars have new assigned (sampled) values
wd, names, values = extract_weights(session) 
print_weights(wd, "NEW ASSIGNMENTS")


# Retrieve protobuf graph definition
graph = tf.get_default_graph()


# [] Uncomment Below if you want to get all operations (long stdout) 
#print("Restored Operations from MetaGraph:")
#for op in graph.get_operations():
#   print(op.name)
   

# Restore operations needed to re-run (aka inference/make predictions)
""" We need this line below for initial state"""
#initial_state = graph.get_tensor_by_name("Test/Model/INITIAL_STATE:0")


#initial_state = graph.get_collection_ref("initial_state")

final_state = graph.get_tensor_by_name("Test/Model/FINAL_STATE:0")
print("Shape final_state: ", final_state.get_shape())

cost = graph.get_tensor_by_name("Test/Model/COST:0")
logits = graph.get_tensor_by_name("Test/Model/LOGITS:0")
print("Shape cost: ", cost.get_shape())
print("Shape logits: ", logits.get_shape())

   
# Run #################################################################


# Global Variables
data_path = "../data"
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, _, id_to_word = raw_data
verbose=True

config = bm.TestConfig()
config.prior_pi = 0.25
config.log_sigma1 =  -1.0
config.log_sigma2 = -7.0

eval_config = config
eval_config.batch_size = 1




###########################################################################

# PROBLEM: WE CANT PASS initial_state
#          so here we try to build it
#          but we can't pass tensors 
#          for feed_dict 

        ### Build the BBB LSTM cells to Pass Zero State ###
"""
graph2 = tf.Graph()
with graph2.as_default():
    init_op = tf.global_variables_initializer()
    sess2 = tf.Session(graph=graph2)
    # Construct prior
    prior = bm.Prior(config.prior_pi, config.log_sigma1, config.log_sigma2)
    
    cells = []
    for i in range(config.num_layers):
        cells.append(bm.BayesianLSTMCell(config.hidden_size, prior, is_training=False,                      
                                  name = "bbb_lstm_hyper{}".format(i),
                                  forget_bias=0.0))
    
    cell = MultiRNNCell(cells, state_is_tuple=True)
    initial_state = cell.zero_state(config.batch_size, bm.data_type())
    a, b = initial_state
    #a0, a1 = a
    #b0, b1 = b
    #initial_state2 = tuple([tuple([a0,a1]), tuple([b0,b1])])
    sess2.run(init_op)
    a = sess2.run(a)
    b = sess2.run(b)
    
initial_state = tuple([a,b])

state = initial_state #sess2.run(initial_state)
print("state type: ", type(state))
"""
###########################################################################


"""
Here we are basically replicating run_epoch() to see if it runs
"""
"""run_epoch()"""

start_time = time.time()
costs = 0.0
iters = 0


### Replicate Zero_State with numpy ###

#a= np.zeros((20,2), dtype=np.float32)
#c = tf.Variable(a)                                                                                                                                                                                                                                                                                                
#h = tf.Variable(a)      
#init = tf.initialize_all_variables() 
#ch = tf.contrib.rnn.LSTMStateTuple(c, h)
#session.run(init)
#ch = session.run(ch)

### Replicate Zero_State with tensor states ###

c = graph.get_tensor_by_name("Test/Model/initial_state_c:0")
h = graph.get_tensor_by_name("Test/Model/initial_state_h:0")
initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

state = initial_state #session.run(initial_state)



fetches = {
    "cost": cost,
    "final_state": final_state,
    }

for step in range(600):#(model.input.epoch_size):
    
    feed_dict = {}
    
    for i, (c, h) in enumerate(initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

    
    #for i, (c,h) in enumerate(initial_state):
    #    feed_dict[0] = state[i][0]
    #    feed_dict[1] = state[i][1]    
    
    vals = session.run(fetches)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += 2 #model.input.num_steps
    
    #Print the first 100 predictions vs targets
    if step <= 100:
        word_logit = session.run(logits)
        single_word = word_logit[0,0,:]
        print("Most probable word is: ",id_to_word[int(np.argmax(single_word))])
        #targets = session.run(model.labels_id)
        #single_target = targets[0,0]
        #print("Target is : ", id_to_word[int(single_target)]) #data[step])
        


    if verbose and (step % (600 // 10) == 10 or step == 0):#(model.input.epoch_size // 10) == 10 or step == 0):
        print("%.3f perplexity: %.3f speed: %.0f wps" %
              (step * 1.0 / 600, np.exp(costs / iters),
               iters * 1 / (time.time() - start_time)))#model.input.batch_size / (time.time() - start_time)))
        #print("Accuracy: ", accuracy)

            
print("We made it! Perplexity: ", np.exp(costs / iters)) #, accuracy



    
if __name__ == '__main__':
    pass
  
    

"""
session.close()




"""
