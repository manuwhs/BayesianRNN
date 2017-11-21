"""
An implementation of the `Local reparameterization trick`
from Kingma & Wellings and 
Bayesian RNN 
from Fortunato, Blundell & Vinyals
"""
import os
import time
import copy
from os.path import join as pjoin
from six.moves import xrange

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from loader import TextLoader, noise_batch

from cfg import PTB_DATA_PATHS, TEXT8_DATA_PATHS, WIKITEXT2_DATA_PATHS

import logging
import sys

logging.basicConfig(level=logging.INFO)

flags = tf.flags

# Settings
flags.DEFINE_integer("hidden_dim", 512, "hidden dimension")
flags.DEFINE_integer("layers", 2, "number of hidden layers")
flags.DEFINE_integer("unroll", 35, "number of time steps to unroll for BPTT")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_float("init_scale", 0.1, "scale for random initialization")
flags.DEFINE_float("bayes_init_scale", 0.05, "scale for random initialization")
flags.DEFINE_float("pi", 0.25, "mixture parameter on the Gaussians")
flags.DEFINE_float("log_sigma1", -1.0, "log sigma for the first mixture comp")
flags.DEFINE_float("log_sigma2", -7.0, "log sigma for the second mixture comp")
flags.DEFINE_float("learning_rate", 1.0, "initial learning rate")
flags.DEFINE_float("learning_rate_decay", 0.9, "amount to decrease learning rate")
flags.DEFINE_float("decay_threshold", 0.0, "decrease learning rate if validation cost difference less than this value")
flags.DEFINE_integer("max_decays", 8, "stop decreasing learning rate after this many times")
flags.DEFINE_float("drop_prob", 0.0, "probability of dropping units")
flags.DEFINE_float("gamma", 0.0, "probability of noising input data")
flags.DEFINE_float("norm_scale", 0.1, "hyperparameter on ixh")
flags.DEFINE_boolean("tied", False, "train with weight tying or not")
flags.DEFINE_integer("epoch", 0, "which epoch to load model from")
flags.DEFINE_boolean("absolute_discounting", False, "scale gamma by absolute discounting factor")
flags.DEFINE_integer("max_epochs", 400, "maximum number of epochs to train")
flags.DEFINE_float("clip_norm", 5.0, "value at which to clip gradients")
flags.DEFINE_string("optimizer", "sgd", "optimizer")
flags.DEFINE_string("run_dir", "sandbox", "directory to store experiment outputs")
flags.DEFINE_string("token_type", "word", "use word or character tokens")
flags.DEFINE_string("scheme", "blank", "use blank or ngram noising scheme")
flags.DEFINE_string("ngram_scheme", "unigram", "use {unigram, uniform, bgkn, mbgkn}")
flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from")
flags.DEFINE_integer("seed", 123, "random seed to use")
flags.DEFINE_integer("steps_per_summary", 10, "how many steps between writing summaries")
flags.DEFINE_boolean("final", False, "final evaluation (run on test after picked best model)")
flags.DEFINE_string("dataset", "ptb", "ptb or text8 or wikitext2")

FLAGS = flags.FLAGS


def get_optimizer(name):
    if name == "sgd":
        return tf.train.GradientDescentOptimizer
    elif name == "adam":
        return tf.train.AdamOptimizer
    else:
        assert False


# Getting stale file handle errors
def log_info(s):
    try:
        logging.info(s)
    except IOError:
        time.sleep(60)


class MixturePrior(object):
    def __init__(self, pi, log_sigma1, log_sigma2):
        self.mean = 0
        self.sigma_mix = pi * tf.exp(log_sigma1) + (1 - pi) * tf.exp(log_sigma2)

    def get_kl_divergence(self, gaussian1):
        # because the other compute_kl does log(sigma) and this is already set
        mean1, sigma1 = gaussian1
        mean2, sigma2 = self.mean, self.sigma_mix

        kl_divergence = tf.log(sigma2) - tf.log(sigma1) + \
                        ((tf.square(sigma1) + tf.square(mean1 - mean2)) / (2 * tf.square(sigma2))) \
                        - 0.5

        return tf.reduce_mean(kl_divergence)

# should only use inside RNN
def get_random_normal_variable(name, mean, prior, shape, dtype):
    """
    A wrapper around tf.get_variable which lets you get a "variable" which is
     explicitly a sample from a normal distribution.
    """

    # Inverse of a softplus function, so that the value of the standard deviation
    # will be equal to what the user specifies, but we can still enforce positivity
    # by wrapping the standard deviation in the softplus function.
    # standard_dev = tf.log(tf.exp(standard_dev) - 1.0) * tf.ones(shape)

    # it's important to initialize variances with care, otherwise the model takes too long to converge
    rho_max_init = tf.log(tf.exp(prior.sigma_mix / 2.0) - 1.0)
    rho_min_init = tf.log(tf.exp(prior.sigma_mix / 4.0) - 1.0)
    std_init = tf.random_uniform_initializer(rho_min_init, rho_max_init)

    # this is constant, original paper/email is not constant
    mean = tf.get_variable(name + "_mean", shape,
                           initializer=tf.constant_initializer(mean),
                           dtype=dtype)

    standard_deviation = tf.get_variable(name + "_standard_deviation", shape,
                                         initializer=std_init,
                                         dtype=dtype)

    standard_deviation = tf.nn.softplus(standard_deviation) + 1e-5
    weights = mean + (standard_deviation * tf.random_normal(shape, 0.0, 1.0, dtype))
    return weights, mean, standard_deviation


class BayesianLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, num_units, prior, is_training=True, forget_bias=1.0, input_size=None, state_is_tuple=True,
                 activation=tf.tanh):

        # once generated they stay the same across time-steps
        # must construct different cell for each layer
        self.is_training = is_training 
        self.prior = prior
        self.W, self.b = None, None

        self.W_mu, self.W_std = None, None
        self.b_mu, self.b_std = None, None

        super(BayesianLSTMCell, self).__init__(num_units, forget_bias, input_size, state_is_tuple, activation)

    # we'll see if this implementation is correct
    def get_W(self, total_arg_size, output_size, dtype):
        with tf.variable_scope("CellWeight"):
            if self.W is None:
                # can use its own init_scale
                self.W, self.W_mu, self.W_std = get_random_normal_variable("Matrix", 0.0, self.prior,
                                                                           [total_arg_size, output_size], dtype=dtype)
        if self.is_training:
            return self.W
        else:
            return self.W_mu

    def get_b(self, output_size, dtype):
        with tf.variable_scope("CellBias"):
            if self.b is None:
                self.b, self.b_mu, self.b_std = get_random_normal_variable("Bias", 0.0, self.prior,
                                                                           [output_size], dtype=dtype)
        if self.is_training:
            return self.b
        else:
            return self.b_mu  # at evaluation time we only do MAP (on mean value)

    def get_kl(self):
        # compute KL divergence internally (more modular code)
        theta_kl = self.prior.get_kl_divergence((self.W_mu, self.W_std))
        theta_kl += self.prior.get_kl_divergence((self.b_mu, self.b_std))

        return theta_kl

    def stochastic_linear(self, args, output_size, bias, bias_start=0.0, scope=None):
        # Local reparameterization trick
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = self.get_W(total_arg_size, output_size, dtype=dtype)
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(1, args), matrix)
            if not bias:
                return res
            bias_term = self.get_b(output_size, dtype=dtype)

        return res + bias_term

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(1, 2, state)
            concat = self.stochastic_linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1, 4, concat)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(1, [new_c, new_h])
            return new_h, new_state


def compute_info_loss(outputs, cell, unroll):
    # outputs: (batch_size, time, hidden_size)
    for time_step in range(unroll):
        pass


class LanguageModel(object):
    def __init__(self, flags, vocab_size, is_training=True):
        batch_size = flags.batch_size
        unroll = flags.unroll
        self._x = tf.placeholder(tf.int32, [batch_size, unroll])
        self._y = tf.placeholder(tf.int32, [batch_size, unroll])
        self._len = tf.placeholder(tf.int32, [None, ])

        in_size = flags.hidden_dim
        prior = MixturePrior(flags.pi, flags.log_sigma1, flags.log_sigma2)

        # use Bayesian LSTM Cell instead
        # under this, epsilon is sampled once in all time steps, and each layer is different
        multi_layer_cell = []
        for _ in range(flags.layers):
            lstm_cell = BayesianLSTMCell(flags.hidden_dim, prior, is_training=is_training,
                                         forget_bias=1.0, state_is_tuple=True)
            multi_layer_cell.append(lstm_cell)

        cell = tf.nn.rnn_cell.MultiRNNCell(multi_layer_cell, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            self.embeddings = tf.get_variable("embeddings", [vocab_size, flags.hidden_dim])
            inputs = tf.nn.embedding_lookup(self.embeddings, self._x)

        # These options (fixed unroll or dynamic_rnn) should give same results but
        # using fixed here since faster

        if True:
            outputs = []
            state = self._initial_state
            with tf.variable_scope("RNN"):
                for time_step in range(unroll):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)
            outputs = tf.reshape(tf.concat(1, outputs), [-1, flags.hidden_dim])

        softmax_w, softmax_w_mu, softmax_w_std = get_random_normal_variable("softmax_w", 0., prior,
                                                                        [flags.hidden_dim, vocab_size], dtype=tf.float32)
        softmax_b, softmax_b_mu, softmax_b_std = get_random_normal_variable("softmax_b", 0., prior,
                                                                            [vocab_size], dtype=tf.float32)

        if is_training:
            logits = tf.matmul(outputs, softmax_w) + softmax_b
        else:
            logits = tf.matmul(outputs, softmax_w_mu) + softmax_b_mu

        seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(logits, [-1, vocab_size])],
            [tf.reshape(self._y, [-1])],
            [tf.ones([batch_size * unroll])])
        # NLL loss
        self.loss = tf.reduce_sum(seq_loss) / batch_size

        # KL loss
        self.kl_loss = 0.0
        for i in range(flags.layers):
            self.kl_loss += multi_layer_cell[i].get_kl()

        self.kl_loss += prior.get_kl_divergence((softmax_w_mu, softmax_w_std))
        self.kl_loss += prior.get_kl_divergence((softmax_b_mu, softmax_b_std))

        # if these don't really work, we can remove input dropout
        # and make softmax projection bayesian as well

        self._final_state = state

        self.ixh = tf.global_norm([outputs])

        if not is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        shapes = [tvar.get_shape() for tvar in tvars]
        log_info("# params: %d" % np.sum([np.prod(s) for s in shapes]))

        total_cost = tf.add(self.loss, self.kl_loss)

        # careful at this part
        grads = tf.gradients(total_cost, tvars)  # + FLAGS.norm_scale * self.ixh
        if flags.clip_norm is not None:
            grads, grads_norm = tf.clip_by_global_norm(grads, flags.clip_norm)
        else:
            grads_norm = tf.global_norm(grads)
        optimizer = get_optimizer(flags.optimizer)(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Summaries for TensorBoard, note this is only within training portion
        with tf.name_scope("summaries"):
            tf.scalar_summary("loss", self.loss / unroll)
            tf.scalar_summary("learning_rate", self.lr)
            tf.scalar_summary("grads_norm", grads_norm)

    def set_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


def run_epoch(epoch_ind, session, model, loader, split, update_op, flags,
              writer=None, summary_op=None, verbose=True):
    """Run an epoch of training/testing"""
    epoch_size = loader.get_num_batches(split)
    start_time = time.time()
    total_cost = 0.0
    total_ixh = 0.0
    state = session.run(model._initial_state)
    iters = 0
    for k in xrange(epoch_size):
        x, y = loader.get_batch(split, k)
        if split == "train":
            gamma = flags.gamma
            x, y = noise_batch(x, y, flags, loader, gamma=gamma)
        seq_len = [y.shape[1]] * flags.batch_size
        fetches = [model.loss, update_op, model.ixh, model._final_state]
        feed_dict = {model._x: x,
                     model._y: y,
                     model._len: seq_len,
                     model._initial_state: state}
        if summary_op is not None and writer is not None:
            fetches = [summary_op] + fetches
            summary, cost, _, ixh, state = session.run(fetches, feed_dict)
            if k % flags.steps_per_summary == 0:
                writer.add_summary(summary, epoch_size * epoch_ind + k)
        else:
            cost, _, ixh, state = session.run(fetches, feed_dict)
        total_cost += cost
        total_ixh += ixh
        iters += flags.unroll

        if k % (epoch_size // 10) == 10 and verbose:
            log_info("%.3f perplexity: %.3f ixh: %.3f speed: %.0f tps" %
                     (k * 1.0 / epoch_size, np.exp(total_cost / iters),
                      total_ixh / iters,
                      iters * flags.batch_size / (time.time() - start_time)))

    return np.exp(total_cost / iters)


def main(_):
    if not os.path.exists(FLAGS.run_dir):
        os.makedirs(FLAGS.run_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.run_dir))
    logging.getLogger().addHandler(file_handler)

    # DATA_PATHS = PTB_DATA_PATHS if FLAGS.dataset == "ptb" else TEXT8_DATA_PATHS
    np.random.seed(FLAGS.seed)

    if FLAGS.dataset == "ptb":
        DATA_PATHS = PTB_DATA_PATHS
    elif FLAGS.dataset == "text8":
        DATA_PATHS = TEXT8_DATA_PATHS
    else:
        DATA_PATHS = WIKITEXT2_DATA_PATHS

    log_info(str(DATA_PATHS))
    data_loader = TextLoader(DATA_PATHS, FLAGS.batch_size, FLAGS.unroll,
                             FLAGS.token_type)
    vocab_size = len(data_loader.token_to_id)
    log_info("Vocabulary size: %d" % vocab_size)
    log_info(FLAGS.__flags)

    eval_flags = copy.deepcopy(FLAGS)
    eval_flags.batch_size = 1
    eval_flags.unroll = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

        # Create training, validation, and evaluation models
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = LanguageModel(FLAGS, vocab_size, is_training=True)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = LanguageModel(FLAGS, vocab_size, is_training=False)
            mtest = LanguageModel(eval_flags, vocab_size, is_training=False)

        summary_op = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.run_dir)
        model_saver = tf.train.Saver(max_to_keep=FLAGS.max_epochs)
        tf.initialize_all_variables().run()

        if FLAGS.restore_checkpoint is not None:
            model_saver.restore(session, FLAGS.restore_checkpoint)

        lr = FLAGS.learning_rate
        decay_count = 0
        prev_valid_perplexity = None
        valid_perplexities = list()

        k = best_epoch = -1

        for k in xrange(FLAGS.max_epochs):
            mtrain.set_lr(session, lr)
            log_info("Epoch %d, learning rate %f" % (k, lr))

            train_perplexity = run_epoch(k, session, mtrain, data_loader, "train",
                                         mtrain._train_op, FLAGS, writer=train_writer, summary_op=summary_op)
            log_info("Epoch: %d Train Perplexity: %.3f" % (k, train_perplexity))
            valid_perplexity = run_epoch(k, session, mvalid, data_loader, "valid",
                                         tf.no_op(), FLAGS, verbose=False)
            log_info("Epoch: %d Valid Perplexity: %.3f" % (k, valid_perplexity))

            if prev_valid_perplexity != None and \
                                    np.log(best_valid_perplexity) - np.log(valid_perplexity) < FLAGS.decay_threshold:
                lr = lr * FLAGS.learning_rate_decay
                decay_count += 1
                log_info("Loading epoch %d parameters, perplexity %f" % \
                         (best_epoch, best_valid_perplexity))
                model_saver.restore(session, pjoin(FLAGS.run_dir, "model_epoch%d.ckpt" % best_epoch))

            prev_valid_perplexity = valid_perplexity

            valid_perplexities.append(valid_perplexity)
            if valid_perplexity <= np.min(valid_perplexities):
                best_epoch = k
                best_valid_perplexity = valid_perplexities[best_epoch]
                save_path = model_saver.save(session, pjoin(FLAGS.run_dir, "model_epoch%d.ckpt" % k))
                log_info("Saved model to file: %s" % save_path)

            if decay_count > FLAGS.max_decays:
                log_info("Reached maximum number of decays, quiting after epoch %d" % k)
                break

        if best_epoch == -1:
            assert FLAGS.epoch != 0
            best_epoch = k = FLAGS.epoch
            best_valid_perplexity = 0

        log_info("Loading epoch %d parameters, perplexity %f" % \
                 (best_epoch, best_valid_perplexity))
        model_saver.restore(session, pjoin(FLAGS.run_dir, "model_epoch%d.ckpt" % best_epoch))
        data_loader = TextLoader(DATA_PATHS, eval_flags.batch_size, eval_flags.unroll, FLAGS.token_type)

        if FLAGS.final:
            test_perplexity = run_epoch(k, session, mtest, data_loader, "test",
                                        tf.no_op(), eval_flags, verbose=False)
            log_info("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()

