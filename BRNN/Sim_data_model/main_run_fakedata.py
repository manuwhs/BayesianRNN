from brnn_model_fake_data import *
import reader

import subprocess
import tensorflow as tf
import import_folders
import pickle_lib as pkl
from graph_lib import gl
import numpy as np
import matplotlib.pyplot as plt
plt.close("all") # Close all previous Windows

"""
    Global variables
"""
model_type = "test"
data_path = "../data/"
save_path = "./saved_model/"
global_prior_pi = 0.25
global_log_sigma1 = -1.0
global_log_sigma2 = -7.0
global_random_seed = 12
global_num_gpus = 1



############## FLAGS ########################
load_config = 0
load_data = 0
build_models = 0
train_models = 0
test_models = 0
plot_data = 1

if (load_config):
    # Model can be "test", "small", "medium", "large"
    model_select = "test"
    model_type = model_select
    #Put the path to the data here
    dat_path = "../data"
    
    #Put the path to where you want to save the training data
    sav_path = "tensorboard/"
    
    # The mixing degree for the prior gaussian mixture
    # As in Fortunato they report scanning
    # mix_pi \in { 1/4, 1/2, 3/4 }
    mixing_pi = 0.25
    
    # As in Fortunato they report scanning
    # log sigma1 \in { 0, -1, -2 }
    # log sigma2 \in { -6, -7, -8 }
    prior_log_sigma1 = -1.0
    prior_log_sigma2 = -7.0
    
    
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
        
        X_dim = 200 # Size of the embedding
    
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
    
        X_dim = 50 # Size of the embedding
        
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
    
        X_dim = 100 # Size of the embedding
        
    class TestConfig(object):
        """Tiny config, for testing."""
        init_scale = 0.1
        learning_rate = 0.5
        max_grad_norm = 1
        num_layers = 2
        num_steps = 20
        hidden_size = 15
        max_epoch = 1
        max_max_epoch = 10
        keep_prob = 1.0
        lr_decay = 0.9
        batch_size = 10
        
        vocab_size = 10000
    
        X_dim = 19 # Size of the embedding
    
    
    #    global_random_seed = set_random_seed
        
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
    
        print ("Model Type")
        print (model_type)
        config.prior_pi = global_prior_pi
        config.log_sigma1 = global_log_sigma1
        config.log_sigma2 = global_log_sigma2
    
        return config

if (load_data):

    print (model_type)
    ####### Global data reading #########
    Ndivisions = 10;
    folder_data = "./data/artificial/"
    
    X_list = pkl.load_pickle(folder_data +"X_values.pkl",Ndivisions)
    Y_list = pkl.load_pickle(folder_data +"Y_values.pkl",Ndivisions)
    t_list = pkl.load_pickle(folder_data +"t_values.pkl",Ndivisions)
    
    num_steps, X_dim = X_list[0].shape
    num_chains = len(X_list)
    
    
    ## Divide in train val and test
    proportion_tr = 0.8
    proportion_val = 0.1
    proportion_tst = 1 -( proportion_val + proportion_tr)
    
    num_tr = 10000
    num_val = 5000
    num_tst = 5000
    
    train_X = [X_list[i] for i in range(num_tr)]
    train_Y = [Y_list[i] for i in range(num_tr)]
    
    val_X = [X_list[i] for i in range(num_tr, num_tr + num_val)]
    val_Y = [Y_list[i] for i in range(num_tr, num_tr + num_val)]
    
    tst_X = [X_list[i] for i in range(num_tr + num_val,  num_tr + num_val + num_tst)]
    tst_Y = [Y_list[i] for i in range(num_tr + num_val,  num_tr + num_val + num_tst)]
    
    # Create the objects with the hyperparameters that will be fed to the network
    train_config = get_config()
    eval_config = get_config( )
    
    ###### Over Set parameters #####
    train_config.X_dim  = X_dim
    eval_config.X_dim  = X_dim
    train_config.num_steps  = num_steps
    eval_config.num_steps  = num_steps
    
    train_config.vocab_size = 2
    eval_config.vocab_size= 2
    
    
    eval_config.batch_size = 2
        
    #eval_config.num_steps = 1
    
    print ("Number of total initial chains %i"%len(X_list))
    print ("Dimensionality of chains (num_step,X_dim)",X_list[0].shape )
    
    ### TODO: Plot the loaded data to check its impurity
    plot_realizations_signal_generated_as_output = 1
    if (plot_realizations_signal_generated_as_output):
        num_plot = 100
        flag = 1;
        legend = ["Realizations"]
        labels = ["Gaussian Process X(t) = mu(t) + e(t)","t", "X(t)"]
        
        for i in range(num_plot):
                gl.plot(t_list[i],X_list[i], lw = 3, ls = "-", alpha = 0.5, nf = flag, legend = legend)
                if (flag == 1):
                    flag = 0
                    legend = []
        
    
if (build_models):
    
    tf.reset_default_graph()
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                                train_config.init_scale)

    with tf.name_scope("Train"):
        train_input = BBB_LSTM_Artificial_Data_Input(batch_size = train_config.batch_size, 
                                                        X = train_X, Y = train_Y,  name="TrainInput")
        
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=train_config, input_=train_input)
        tf.summary.scalar("Training_Loss", m.cost)
        tf.summary.scalar("Learning_Rate", m.lr)
        tf.summary.scalar("KL Loss", m.kl_loss)
        tf.summary.scalar("Total Loss", m.total_loss)

    print ("Creating Validation model")
    with tf.name_scope("Valid"):
        valid_input = BBB_LSTM_Artificial_Data_Input(batch_size = eval_config.batch_size, 
                                                            X = val_X, Y = val_Y,  name="ValidInput")
        
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=eval_config, input_=valid_input)
        tf.summary.scalar("Validation_Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = BBB_LSTM_Artificial_Data_Input(batch_size = eval_config.batch_size, 
                                                            X = tst_X, Y = tst_Y,  name="TestInput")
            
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config,
                             input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
        model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    soft_placement = False
    if global_num_gpus > 1:
        soft_placement = True
        util.auto_parallel(metagraph, m)

if (train_models):
    
## Training !
    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
    
            for i in range(train_config.max_max_epoch):
                lr_decay = train_config.lr_decay ** max(i + 1 - train_config.max_epoch, 0.0)
                m.assign_lr(session, train_config.learning_rate * lr_decay)
    
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                
            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
            
    
            print("Saving model to %s." % save_path)
            sv.saver.save(session, save_path, global_step=sv.global_step)

if (test_models):
    ## Testing
    print ("Testing")
    predicted = []   # Variable to store predictions
    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            
           # session = tf.Session()
        
            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
    
            print ("----------------------------------------------------------------")
            print ("------------------ Prediction of Output ---------------------")
    
           #  inputs, predicted = fetch_output(session, mtest)
    
            costs = 0.0
            state = session.run(model.initial_state)
    
            inputs = []
            outputs = []
            targets = []
            fetches = {
                "final_state": model.final_state,
                "output": model.output,
                "input": model.input_data,
                "targets": model.targets
            }
    
            for step in range(model.input.epoch_size):
                feed_dict = {}
                for i, (c, h) in enumerate(model.initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h
    
                print ("Computing batch %i/%i"%(step, model.input.epoch_size))
                vals = session.run(fetches, feed_dict)
                state = vals["final_state"]
                output = vals["output"]
                input_i = vals["input"]
                
                outputs.append(output)
                inputs.append(input_i)
                targets.append(vals["targets"])
                if (step == 100):
                    break;

if (plot_data):
    
    batch_i = 5
    data = np.array(inputs[batch_i][0])[:,[0]]
    labels = np.array(targets[batch_i][0])[:]
    predicted = np.array(outputs[batch_i][0])[:,[1]]
    #print(data)
    #print(labels)
    #print (predicted)
    
    labels_chart = ["Example output for medium noise level", "","X[n]"]
    gl.set_subplots(3,1)
    ax1 = gl.plot(np.array(range(data.size)), data, nf = 1, labels = labels_chart, legend = ["X[n]"])
    ax2 = gl.stem(np.array(range(data.size)),labels, nf = 1, sharex = ax1, labels = ["","","Y[n]"], bottom = 0.5,
                  legend = ["Targets Y[n]"])
    gl.stem(np.array(range(data.size)),predicted, nf = 1, sharex = ax1, sharey = ax2, labels = ["","n","O[n]"],bottom = 0.5,
            legend = ["Predictions O[n]"])

    gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.2, hspace=0.01)