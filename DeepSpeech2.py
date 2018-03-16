#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

import datetime
import pickle
import shutil
import subprocess
import tensorflow as tf
import time
import traceback
import inspect
import math


from six.moves import zip, range, filter, urllib, BaseHTTPServer
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.tools import freeze_graph
from threading import Thread, Lock
from util.audio import audiofile_to_input_vector
from util.feeding import DataSet, ModelFeeder
from util.gpu import get_available_gpus, get_available_cpus
from util.shared_lib import check_cupti
from util.text import sparse_tensor_value_to_texts, wer, Alphabet, ndarray_to_text, levenshtein
from xdg import BaseDirectory as xdg
import numpy as np

# Importer
# ========

tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')
tf.app.flags.DEFINE_boolean ('fulltrace',        False,       'if full trace debug info should be generated during training')
tf.app.flags.DEFINE_boolean ('train_sort',       False,       'sort training set by wave length')


# Cluster configuration
# =====================

tf.app.flags.DEFINE_string  ('ps_hosts',         '',          'parameter servers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('worker_hosts',     '',          'workers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('job_name',         'localhost', 'job name - one of localhost (default), worker, ps')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of task within the job - worker with index 0 will be the chief')
tf.app.flags.DEFINE_integer ('replicas',         -1,          'total number of replicas - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('replicas_to_agg',  -1,          'number of replicas to aggregate - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('coord_retries',    100,         'number of tries of workers connecting to training coordinator before failing')
tf.app.flags.DEFINE_string  ('coord_host',       'localhost', 'coordination server host')
tf.app.flags.DEFINE_integer ('coord_port',       2500,        'coordination server port')
tf.app.flags.DEFINE_integer ('iters_per_worker', 1,           'number of train or inference iterations per worker before results are sent back to coordinator')

# Global Constants
# ================

tf.app.flags.DEFINE_boolean ('train',            True,        'wether to train the network')
tf.app.flags.DEFINE_boolean ('test',             True,        'wether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            100,         'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_float   ('dropout_keep_prob',  1.00,        'dropout keep probability')

tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters
tf.app.flags.DEFINE_string  ('optimizer',        'adam',      'optimizer type: adam, momentum')
tf.app.flags.DEFINE_float   ('beta1',            0.9,         'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('beta2',            0.999,       'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.0001,       'learning rate of Adam optimizer')

# SGD with momentum optimizer
tf.app.flags.DEFINE_integer ('decay_steps',      0,           'number of LR decay steps')
tf.app.flags.DEFINE_float   ('decay_rate',       0.1,         'decay_rate')
tf.app.flags.DEFINE_float   ('momentum',         0.9,         'momentum for SGD with momentum')

tf.app.flags.DEFINE_float   ('weight_decay',     0.0,         'weight_decay')


# Batch sizes

tf.app.flags.DEFINE_integer ('train_batch_size', 1,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,           'number of elements in a test batch')

# Sample limits

tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set - 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set- 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set- 0 means no limit')

# Step widths

tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model - a detailed progress report is dependent on "--display_step" - 0 means no validation steps')

# Checkpointing

tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  72000,         'checkpoint saving interval in seconds')
tf.app.flags.DEFINE_integer ('checkpoint_steps',   1,         'checkpoint saving interval in epochs')
tf.app.flags.DEFINE_integer ('max_to_keep',      5,           'number of checkpoint files to keep - default value is 5')

# Exporting

tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'wether to remove old exported models')
tf.app.flags.DEFINE_boolean ('use_seq_length',   True,        'have sequence_length in the exported graph (will make tfcompile unhappy)')

# Reporting

tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')

tf.app.flags.DEFINE_string  ('wer_log_pattern',  '',          'pattern for machine readable global logging of WER progress; has to contain %%s, %%s and %%f for the set name, the date and the float respectively; example: "GLOBAL LOG: logwer(\'12ade231\', %%s, %%s, %%f)" would result in some entry like "GLOBAL LOG: logwer(\'12ade231\', \'train\', \'2017-05-18T03:09:48-0700\', 0.05)"; if omitted (default), there will be no logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'wether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     None,        'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')
tf.app.flags.DEFINE_integer ('summary_steps',    None,        'interval in steps for saving TensorBoard summaries - if 0, no summaries will be written')

# Data augmentation
tf.app.flags.DEFINE_boolean ('augment',          False,         'augment training dataset')
tf.app.flags.DEFINE_float   ('time_stretch_ratio', 0.2,        '+/- time_stretch_ratio speed up / slow down')
tf.app.flags.DEFINE_integer ('noise_level_min', -90,           'minimum level of noise, dB')
tf.app.flags.DEFINE_integer ('noise_level_max', -46,            'maximum level of noise, dB')

# Geometry
tf.app.flags.DEFINE_string  ('input_type',       'spectrogram','input features type: mfcc or spectrogram')
tf.app.flags.DEFINE_integer ('num_audio_features',  161,       'number of mfcc coefficients or spectrogram frequency bins')
tf.app.flags.DEFINE_integer ('num_pad', 0, 'padding from both side of sequence')

tf.app.flags.DEFINE_integer ('num_conv_layers',  2,            'layer width to use when initialising layers')
tf.app.flags.DEFINE_integer ('num_rnn_layers',   1,            'layer width to use when initialising layers')
tf.app.flags.DEFINE_string  ('rnn_type',        'gru',         'rnn-cell type')
tf.app.flags.DEFINE_integer ('rnn_cell_dim',     1024,         'rnn-cell dim')
tf.app.flags.DEFINE_boolean ('rnn_unidirectional', False,      'bi-directional or uni-directional')

tf.app.flags.DEFINE_boolean ('row_conv',          False,      'row convolution')
tf.app.flags.DEFINE_integer ('row_conv_width',     16,         'rnn-cell dim')

tf.app.flags.DEFINE_integer ('n_hidden',         1024,        'layer width to use when initialising layers')

# Initialization

tf.app.flags.DEFINE_integer ('random_seed',      1,           'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.0001,      'default standard deviation to use when initialising weights and biases')

# Early Stopping

tf.app.flags.DEFINE_boolean ('early_stop',       False,        'enable early stopping mechanism over validation dataset. Make sure that dev FLAG is enabled for this to work')

# This parameter is irrespective of the time taken by single epoch to complete and checkpoint saving intervals.
# It is possible that early stopping is triggered far after the best checkpoint is already replaced by checkpoint saving interval mechanism.
# One has to align the parameters (earlystop_nsteps, checkpoint_secs) accordingly as per the time taken by an epoch on different datasets.

tf.app.flags.DEFINE_integer ('earlystop_nsteps',  4,          'number of steps to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
tf.app.flags.DEFINE_float   ('estop_mean_thresh', 0.5,        'mean threshold for loss to determine the condition if early stopping is required')
tf.app.flags.DEFINE_float   ('estop_std_thresh',  0.5,        'standard deviation threshold for loss to determine the condition if early stopping is required')

# Decoder

tf.app.flags.DEFINE_string  ('decoder_library_path', 'native_client/libctc_decoder_with_kenlm.so', 'path to the libctc_decoder_with_kenlm.so library containing the decoder implementation.')
tf.app.flags.DEFINE_string  ('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
tf.app.flags.DEFINE_string  ('lm_binary_path',       'data/lm/lm.binary', 'path to the language model binary file created with KenLM')
tf.app.flags.DEFINE_string  ('lm_trie_path',         'data/lm/trie', 'path to the language model trie file created with native_client/generate_trie')
tf.app.flags.DEFINE_integer ('beam_width',        100,   'beam width used in the CTC decoder when building candidate transcriptions')
tf.app.flags.DEFINE_float   ('lm_weight',         1.0,  'the alpha hyperparameter of the CTC decoder. Language Model weight.')
tf.app.flags.DEFINE_float   ('word_count_weight', 1.50,  'the beta hyperparameter of the CTC decoder. Word insertion weight (penalty).')
tf.app.flags.DEFINE_float   ('valid_word_count_weight', 2.50, 'valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary.')

# Inference mode

tf.app.flags.DEFINE_string  ('one_shot_infer',       '',       'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it. Disables training, testing and exporting.')

#for var in ['b1', 'w1', 'b2','w2', 'b3', 'w3', 'a2','h2', 'a3', 'h3', 'b5', 'h5', 'b6', 'h6']:
#    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)

FLAGS = tf.app.flags.FLAGS

def initialize_globals():

    # ps and worker hosts required for p2p cluster setup
    FLAGS.ps_hosts = list(filter(len, FLAGS.ps_hosts.split(',')))
    FLAGS.worker_hosts = list(filter(len, FLAGS.worker_hosts.split(',')))

    # Determine, if we are the chief worker
    global is_chief
    is_chief = len(FLAGS.worker_hosts) == 0 or (FLAGS.task_index == 0 and FLAGS.job_name == 'worker')

    # Initializing and starting the training coordinator
    global COORD
    COORD = TrainingCoordinator()
    COORD.start()

    # The absolute number of computing nodes - regardless of cluster or single mode
    global num_workers
    num_workers = max(1, len(FLAGS.worker_hosts))

    # Create a cluster from the parameter server and worker hosts.
    global cluster
    cluster = tf.train.ClusterSpec({'ps': FLAGS.ps_hosts, 'worker': FLAGS.worker_hosts})

    # If replica numbers are negative, we multiply their absolute values with the number of workers
    if FLAGS.replicas < 0:
        FLAGS.replicas = num_workers * -FLAGS.replicas
    if FLAGS.replicas_to_agg < 0:
        FLAGS.replicas_to_agg = num_workers * -FLAGS.replicas_to_agg

    # The device path base for this node
    global worker_device
    worker_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task_index)

    # This node's CPU device
    global cpu_device
    cpu_device = worker_device + '/cpu:0'

    # This node's available GPU devices
    global available_devices
    available_devices = [worker_device + gpu for gpu in get_available_gpus()]
    num_gpus=len(available_devices)
    print("Found %d GPUs" % num_gpus)

    global available_cpus
    available_cpus=get_available_cpus()
    num_cpus = len(available_cpus)
    print("Found %d CPUs" % num_cpus)

    # If there is no GPU available, we fall back to CPU based operation
    if 0 == len(available_devices):
        print("Warning: no GPU found, switch to CPU")
        available_devices = [cpu_device]

    # Set default checkpoint dir
    if len(FLAGS.checkpoint_dir) == 0:
        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech','checkpoints'))

    # Set default summary dir
    if len(FLAGS.summary_dir) == 0:
        FLAGS.summary_dir = xdg.save_data_path(os.path.join('deepspeech','summaries'))

    # Standard session configuration that'll be used for all new sessions.
    global session_config
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_placement)

    global alphabet
    alphabet = Alphabet(os.path.abspath(FLAGS.alphabet_config_path))

    # Geometric Constants
    # ===================

    # For an explanation of the meaning of the geometric constants, please refer to
    # doc/Geometry.md

    # 'mfcc' or 'spectrogram'
    global input_type
    input_type = FLAGS.input_type

    global augmentation_parameters
    augmentation_parameters = {'time_stretch_ratio': FLAGS.time_stretch_ratio,
                               'noise_level_min': FLAGS.noise_level_min,
                               'noise_level_max': FLAGS.noise_level_max}

    # Number of MFCC features or spectrogram frequency bins
    global n_input
    n_input = FLAGS.num_audio_features #176 for 3 rnn # 126 # TODO: Determine this programatically from the sample rate

    # The number of frames in the context
    global n_context
    n_context = 0 # TODO: clean it out

    global n_pad
    n_pad = FLAGS.num_pad
    print("padding: {}".format(n_pad))

    # Number of units in hidden layers

    global num_conv_layers
    num_conv_layers = FLAGS.num_conv_layers

    global conv_layers

    conv_layers = [
        {'kernel_size': [11,41], 'stride': [2,2], 'num_channels': 32, 'padding': 'SAME' },
        {'kernel_size': [11,21], 'stride': [1,2], 'num_channels': 64, 'padding': 'SAME' },
        {'kernel_size': [11,21], 'stride': [1,2], 'num_channels': 96, 'padding': 'SAME'}
    ]
    '''
    conv_layers = [
        {'kernel_size': [3,3], 'stride': [1,1], 'num_channels':  32, 'padding': 'SAME'},
        {'kernel_size': [3,3], 'stride': [2,2], 'num_channels':  48, 'padding': 'SAME' },
        {'kernel_size': [3,3], 'stride': [1,1], 'num_channels':  64, 'padding': 'SAME'},
        {'kernel_size': [3,3], 'stride': [2,2], 'num_channels':  80, 'padding': 'SAME' },
        {'kernel_size': [3,3], 'stride': [1,1], 'num_channels':  96, 'padding': 'SAME' },
        {'kernel_size': [3,3], 'stride': [2,2], 'num_channels': 128, 'padding': 'SAME'}
    ]
    '''
    global reduction_factor
    reduction_factor = 1

    N_conv = min(len(conv_layers), num_conv_layers)
    for i in range(N_conv):
        reduction_factor = reduction_factor * conv_layers[i]['stride'][0]
    print("reduction_factor: {}".format(reduction_factor))

    global num_rnn_layers
    num_rnn_layers = FLAGS.num_rnn_layers

    global rnn_unidirectional
    rnn_unidirectional=FLAGS.rnn_unidirectional

    global n_hidden
    n_hidden = FLAGS.n_hidden

    global weight_decay
    weight_decay = FLAGS.weight_decay

    # The number of characters in the target language plus one
    global n_character
    n_character = alphabet.size() + 1 # +1 for CTC blank label

    # Queues that are used to gracefully stop parameter servers.
    # Each queue stands for one ps. A finishing worker sends a token to each queue befor joining/quitting.
    # Each ps will dequeue as many tokens as there are workers before joining/quitting.
    # This ensures parameter servers won't quit, if still required by at least one worker and
    # also won't wait forever (like with a standard `server.join()`).
    global done_queues
    done_queues = []
    for i, ps in enumerate(FLAGS.ps_hosts):
        # Queues are hosted by their respective owners
        with tf.device('/job:ps/task:%d' % i):
            done_queues.append(tf.FIFOQueue(1, tf.int32, shared_name=('queue%i' % i)))

    # Placeholder to pass in the worker's index as token
    global token_placeholder
    token_placeholder = tf.placeholder(tf.int32)

    # Enqueue operations for each parameter server
    global done_enqueues
    done_enqueues = [queue.enqueue(token_placeholder) for queue in done_queues]

    # Dequeue operations for each parameter server
    global done_dequeues
    done_dequeues = [queue.dequeue() for queue in done_queues]

    if len(FLAGS.one_shot_infer) > 0:
        FLAGS.train = False
        FLAGS.test = False
        FLAGS.export_dir = ''
        if not os.path.exists(FLAGS.one_shot_infer):
            log_error('Path specified in --one_shot_infer is not a valid file.')
            exit(1)


# Logging functions
# =================

def prefix_print(prefix, message):
    print(prefix + ('\n' + prefix).join(message.split('\n')))

def log_debug(message):
    if FLAGS.log_level == 0:
        prefix_print('D ', message)

def log_traffic(message):
    if FLAGS.log_traffic:
        log_debug(message)

def log_info(message):
    if FLAGS.log_level <= 1:
        prefix_print('I ', message)

def log_warn(message):
    if FLAGS.log_level <= 2:
        prefix_print('W ', message)

def log_error(message):
    if FLAGS.log_level <= 3:
        prefix_print('E ', message)


# Graph Creation
# ==============

def variable_on_worker_level(name, shape, initializer, trainable=True, regularizer=None):
    r'''
    an utility function ``variable_on_worker_level() to create a variable in CPU memory.
    '''
    # Use the /cpu:0 device on worker_device for scoped operations
    if len(FLAGS.ps_hosts) == 0:
        device = worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)

    with tf.device(device):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=trainable,
                              regularizer=regularizer)
    return var

# =========================================================

def batch_norm(name, input, training, momentum=0.90):
    n_channels = input.get_shape()[-1]
    variance_epsilon = 0.001
    beta = variable_on_worker_level(name + '/beta',  shape=[n_channels],
                   initializer=tf.zeros_initializer,
                   trainable=True,
                   regularizer=None)
    gamma = variable_on_worker_level(name + '/gamma',  shape=[n_channels],
                   initializer=tf.ones_initializer,
                   trainable=True,
                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay) if (weight_decay > 0.0) else None)

    global_mean = variable_on_worker_level(name + '/g_mean', shape=[n_channels],
                                      initializer= tf.zeros_initializer,
                                      trainable=False,
                                      regularizer=None)
    global_var = variable_on_worker_level(name + '/g_var', shape=[n_channels],
                                     initializer= tf.ones_initializer,
                                     trainable=False,
                                     regularizer=None)

    def bn_train():
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])
        train_mean= tf.assign(global_mean, global_mean*momentum + batch_mean*(1-momentum))
        train_var = tf.assign(global_var,  global_var*momentum + batch_var*(1-momentum))
    #    train_mean = tf.assign(global_mean,  batch_mean )
    #    train_var = tf.assign(global_var,  batch_var)

        with tf.control_dependencies([train_mean, train_var]):
            output = tf.nn.batch_normalization(x=input,
                                           mean=batch_mean, variance=batch_var,
                                           offset=beta, scale=gamma,
                                           variance_epsilon=variance_epsilon)
        return output

    def bn_val():
        output = tf.nn.batch_normalization(x=input,
                                           mean=global_mean, variance=global_var,
                                           offset=beta, scale=gamma,
                                           variance_epsilon=variance_epsilon)
        return output

    output = tf.cond(tf.equal(training, 0), bn_train, bn_val)
    return output

# ============================================================
def conv2D(name,
           input,
           in_channels = 1,
           output_channels = 1,
           kernel_size=[3,3],
           strides=[1,1],
           padding='SAME',
#           activation_fn=lambda x: tf.minimum(tf.nn.relu(x), FLAGS.relu_clip),
           activation_fn=lambda x: tf.nn.relu(x),
           weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
           bias_initializer=tf.zeros_initializer(),
           training = True
           ):

    w = variable_on_worker_level(name+'/w',
                   shape = [kernel_size[0], kernel_size[1], in_channels, output_channels],
                   initializer=weights_initializer,
                   trainable=True,
                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay) if (weight_decay > 0.) else None)

    s = [1, strides[0], strides[1], 1]
    y = tf.nn.conv2d(input, w, s, padding)
    y = batch_norm(name+'/bn', y, training = training)
    '''
    b = variable_on_worker_level(name+'/b',
                   shape=[output_channels],
                   initializer=bias_initializer, 
                   trainable=True, regularizer= None)
    y = tf.nn.bias_add(y, b)
    '''
    output = activation_fn(y)
    return output

# ==========================================================
def row_conv(name, input, batch, channels, width,
           activation_fn=lambda x: tf.nn.relu(x),
           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
           training = True
           ):

    x=tf.reshape(input, [batch,-1,1,channels])

    w = variable_on_worker_level(name+'/w',
                    shape = [width, 1, channels, 1],
                   initializer=weights_initializer, trainable=True,
                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay) if (weight_decay > 0.) else None)
    y = tf.nn.depthwise_conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
    y = batch_norm(name + '/bn', y, training=training)
    y = activation_fn(y)

    output = tf.reshape(y, [batch,-1,channels])
    return output

#===========================================================

def rnn_cell(rnn_cell_dim, layer_type, dropout_keep_prob=1.0):
    reuse=tf.get_variable_scope().reuse
    if (layer_type=="layernorm_lstm"):
         cell = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_cell_dim, forget_bias=1.0,
                        dropout_keep_prob=dropout_keep_prob, dropout_prob_seed=FLAGS.random_seed,
                        reuse=reuse)
    else:
        if (layer_type=="lstm"):
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_cell_dim, reuse=reuse)
        elif (layer_type=="gru"):
            cell = tf.contrib.rnn.GRUCell(rnn_cell_dim, reuse=reuse)
        elif (layer_type=="cudnn_gru"):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_cell_dim, reuse=reuse)
        elif (layer_type == "cudnn_lstm"):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_cell_dim, reuse=reuse)
        else:
            print("Error: not supported rnn type:{}".format(layer_type))

        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob, seed=FLAGS.random_seed)

    return cell

# =========================================================

def DeepSpeech2(batch_x, seq_length, training):
    r'''
    DeepSpeech2-like model https://arxiv.org/pdf/1512.02595.pdf
    '''
    #print(type(training))
    #print(training)

    #dropout_keep_prob = FLAGS.dropout_keep_prob if training else 1.0
 #   dropout_keep_prob = tf.cond(tf.equal(training,tf.constant(0)), lambda: FLAGS.dropout_keep_prob, lambda: 1.0)
    dropout_keep_prob = tf.cond(tf.equal(training, 0), lambda: FLAGS.dropout_keep_prob, lambda: 1.0)

    # Input shape: [B, T, F]
    batch_x_shape = tf.shape(batch_x)
    batch_size = batch_x_shape[0]
    T = batch_x_shape[1]
    F = batch_x_shape[2]
    print("ds2 inputs:", batch_x.get_shape().as_list())
    # Reshaping `batch_x` to a tensor with shape [B, T, F, 1]
    batch_4d = tf.expand_dims(batch_x, dim=-1)
    # print(batch_4d.get_shape()

    #----- Convolutional layers -----------------------------------------------
    '''
    conv_layers = [
        {'kernel_size': [11,41], 'stride': [2,2], 'num_channels': 32, 'padding': 'SAME' },
        {'kernel_size': [11,21], 'stride': [2,2], 'num_channels': 64, 'padding': 'SAME' },
        {'kernel_size': [11,21], 'stride': [2,2], 'num_channels': 96, 'padding': 'SAME'}
    ]
    
    conv_layers = [
        {'kernel_size': [3,3], 'stride': [1,1], 'num_channels':  32, 'padding': 'SAME'},
        {'kernel_size': [3,3], 'stride': [2,2], 'num_channels':  48, 'padding': 'SAME' },
        {'kernel_size': [3,3], 'stride': [1,1], 'num_channels':  64, 'padding': 'SAME'},
        {'kernel_size': [3,3], 'stride': [2,2], 'num_channels':  80, 'padding': 'SAME' },
        {'kernel_size': [3,3], 'stride': [1,1], 'num_channels':  96, 'padding': 'SAME' },
        {'kernel_size': [3,3], 'stride': [2,2], 'num_channels': 128, 'padding': 'SAME'}
    ]
    '''

    # Number of convolutional layers
    N_conv = min(len(conv_layers), num_conv_layers)

    ch_out = batch_4d.get_shape()[-1]
    f_out = n_input
    conv = batch_4d
    for idx_conv in range(N_conv):
        name = 'conv{}'.format(idx_conv+1)
        ch_in = ch_out
        ch_out = conv_layers[idx_conv]['num_channels']
        kernel_size = conv_layers[idx_conv]['kernel_size']  #[time, freq]
        strides = conv_layers[idx_conv]['stride']

        if conv_layers[idx_conv]['padding']== "VALID":
            seq_length = (seq_length - kernel_size[0] + strides[0]) // strides[0]
            f_out = (f_out - kernel_size[1]+strides[1]) // strides[1]
        else:
            seq_length = (seq_length + strides[0] - 1) // strides[0]
            f_out = (f_out + strides[1] -  1) // strides[1]
        print('{}: kernel={} stride={} ch=[{}, {}] f_out={}'.format(
            name, kernel_size, strides, ch_in, ch_out, f_out))

        conv = conv2D(name, input=conv, in_channels=ch_in, output_channels=ch_out,
                   kernel_size=kernel_size, strides=strides,
                   padding=conv_layers[idx_conv]['padding'],
                   # activation_fn=tf.nn.relu,
                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                   bias_initializer=tf.constant_initializer(0.000001),
                   training=training)
    
    # reshape: [B, T, F, C] --> [B, T, FxC]
    B = batch_x_shape[0]
    T = conv.get_shape().as_list()[1]
    F = conv.get_shape().as_list()[2]
    C = conv.get_shape().as_list()[3]
    fc = F*C
    outputs = tf.reshape(conv, [B, -1, fc])

    # ----- RNN ---------------------------------------------------------------
    if (num_rnn_layers > 0):
        rnn_input = outputs
        rnn_cell_dim = int(FLAGS.rnn_cell_dim)
        if rnn_unidirectional:
            print("Uni-directional RNN: num_layers={}, type={}, dim={}".format(num_rnn_layers,
                                                                         FLAGS.rnn_type,rnn_cell_dim))
            multirnn_cell_fw = tf.contrib.rnn.MultiRNNCell(
                [rnn_cell(rnn_cell_dim=rnn_cell_dim, layer_type=FLAGS.rnn_type, dropout_keep_prob=dropout_keep_prob)
                 for _ in    range(num_rnn_layers)])

            outputs, output_states = tf.nn.dynamic_rnn(cell=multirnn_cell_fw,
                                                       inputs=rnn_input,
                                                       sequence_length=seq_length,
                                                       dtype=tf.float32,
                                                       time_major=False
                                                       )
        else:
            print("Bi-directional RNN: num_layers={}, type={}, dim={}".format(num_rnn_layers,
                                                                        FLAGS.rnn_type,rnn_cell_dim))
            multirnn_cell_fw = tf.contrib.rnn.MultiRNNCell(
                [rnn_cell(rnn_cell_dim=rnn_cell_dim, layer_type=FLAGS.rnn_type, dropout_keep_prob=dropout_keep_prob)
                 for _ in range(num_rnn_layers)])
            multirnn_cell_bw = tf.contrib.rnn.MultiRNNCell(
                [rnn_cell(rnn_cell_dim=rnn_cell_dim, layer_type=FLAGS.rnn_type, dropout_keep_prob=dropout_keep_prob)
                 for _ in range(num_rnn_layers)])

            outputs,output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=multirnn_cell_fw, cell_bw= multirnn_cell_bw,
                                                        inputs=rnn_input,
                                                        sequence_length=seq_length,
                                                        dtype=tf.float32,
                                                        time_major=False
                                                        )
            # reshape: 2 x [B, T, n_cell_dim] --> [B, T, 2*n_cell_dim]
            outputs = tf.concat(outputs, 2)

    #----Row-conv----------------------------------------------------------
    if (FLAGS.row_conv and FLAGS.row_conv_width > 1):
        print("Row convolution width={}".format(FLAGS.row_conv_width))
        C = outputs.get_shape().as_list()[-1]
        outputs = row_conv(name="row_conv", input=outputs,
                       batch=batch_x_shape[0], channels=C, width=FLAGS.row_conv_width,
                       training=training)

    #---- reshape: [B, T, C] --> [T, B, C] --------------------------------
    outputs =  tf.transpose(outputs, [1, 0, 2])

    #--- hidden layer with clipped RELU activation and dropout-------------
    n_hidden_in = outputs.get_shape().as_list()[-1]
    outputs = tf.reshape(outputs, [-1, n_hidden_in])
    # fc1=[h5,b5]
    fc1_w = variable_on_worker_level('fc1/w', [n_hidden_in, n_hidden],
                   tf.contrib.layers.xavier_initializer(uniform=True),
                   trainable=True,
                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay) if (weight_decay > 0.) else None)
    fc1_b = variable_on_worker_level('fc1/b', [n_hidden],
                   tf.constant_initializer(0.),
                   trainable=True, regularizer= None)
    outputs = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, fc1_w), fc1_b)), FLAGS.relu_clip)
    outputs = tf.nn.dropout(outputs, dropout_keep_prob)

    #--- logits --------------------------------------------------
    #n_hidden = outputs.get_shape().as_list()[-1]
    fc2_w = variable_on_worker_level('fc2/w', [n_hidden, n_character],
                   tf.contrib.layers.xavier_initializer(uniform=True),
                   trainable=True,
                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay) if (weight_decay > 0.) else None)
    fc2_b = variable_on_worker_level('fc2/b', [n_character],
                   tf.constant_initializer(0.),
                   trainable=True, regularizer= None)
    outputs = tf.add(tf.matmul(outputs, fc2_w), fc2_b)
    #--- reshape: [T*B,A] --> [T, B, A] ---------------------------
    # output format: [n_steps, batch_size, n_character]
    logits = tf.reshape(outputs, [-1, batch_x_shape[0], n_character], name="logits")

    return logits, seq_length
#================================================================================

if not os.path.exists(os.path.abspath(FLAGS.decoder_library_path)):
    print('ERROR: The decoder library file does not exist. Make sure you have ' \
          'downloaded or built the native client binaries and pass the ' \
          'appropriate path to the binaries in the --decoder_library_path parameter.')

custom_op_module = tf.load_op_library(FLAGS.decoder_library_path)

def decode_with_lm(inputs, sequence_length, beam_width=128,
                   top_paths=1, merge_repeated=True):
  decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
      custom_op_module.ctc_beam_search_decoder_with_lm(
          inputs, sequence_length, beam_width=beam_width,
          model_path=FLAGS.lm_binary_path, trie_path=FLAGS.lm_trie_path, alphabet_path=FLAGS.alphabet_config_path,
          lm_weight=FLAGS.lm_weight, word_count_weight=FLAGS.word_count_weight, valid_word_count_weight=FLAGS.valid_word_count_weight,
          top_paths=top_paths, merge_repeated=merge_repeated))

  return (
      [tf.SparseTensor(ix, val, shape) for (ix, val, shape)
       in zip(decoded_ixs, decoded_vals, decoded_shapes)],
      log_probabilities)


def mask_nans(x):
   x_zeros = tf.zeros_like(x)
   x_mask  = tf.is_finite(x)
   y = tf.where(x_mask, x, x_zeros)
   return y

# ===============================================================

# Accuracy and CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).

def calculate_mean_edit_distance_and_loss(model_feeder, tower, training):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_x, batch_seq_len, batch_y, mode = model_feeder.next_batch(tower)

    # Calculate the logits of the batch using BiRNN
 #   logits, batch_seq_len = DeepSpeech2(batch_x, tf.to_int64(batch_seq_len), dropout)

    logits, batch_seq_len = DeepSpeech2(batch_x, batch_seq_len, training=mode)

    # Compute the CTC loss
    total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len,
                                    ignore_longer_outputs_than_inputs = True)

    # check for inf and nans
    total_loss = mask_nans(total_loss)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Beam search decode the batch
    decoded, _ = decode_with_lm(logits, batch_seq_len, merge_repeated=False, beam_width=FLAGS.beam_width)

    # Compute the edit (Levenshtein) distance
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)

    # Compute the mean edit distance
    mean_edit_distance = tf.reduce_mean(distance)

    # Finally we return the
    # - calculated total and
    # - average losses,
    # - the Levenshtein distance,
    # - the recognition mean edit distance,
    # - the decoded batch and
    # - the original batch_y (which contains the verified transcriptions).
    return total_loss, avg_loss, distance, mean_edit_distance, decoded, batch_y


# =======Optimization==========================================================

# Support
# * SGD with momentum (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# * Adam (http://arxiv.org/abs/1412.6980),

def create_optimizer(optimizer='adam' , lr=FLAGS.learning_rate):
    if (optimizer == 'adam'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                               beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon=FLAGS.epsilon)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=FLAGS.momentum)
    return optimizer

# Towers
# ======

# In order to properly make use of multiple GPU's, one must introduce new abstractions,
# not present when using a single GPU, that facilitate the multi-GPU use case.
# In particular, one must introduce a means to isolate the inference and gradient
# calculations on the various GPU's.
# The abstraction we intoduce for this purpose is called a 'tower'.
# A tower is specified by two properties:
# * **Scope** - A scope, as provided by `tf.name_scope()`,
# is a means to isolate the operations within a tower.
# For example, all operations within 'tower 0' could have their name prefixed with `tower_0/`.
# * **Device** - A hardware device, as provided by `tf.device()`,
# on which all operations within the tower execute.
# For example, all operations of 'tower 0' could execute on the first GPU `tf.device('/gpu:0')`.

def get_tower_results(model_feeder, optimizer, training):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate

    * The CTC decodings ``decoded``,
    * The (total) loss against the outcome (Y) ``total_loss``,
    * The loss averaged over the whole batch ``avg_loss``,
    * The optimization gradient (computed based on the averaged loss),
    * The Levenshtein distances between the decodings and their transcriptions ``distance``,
    * The mean edit distance of the outcome averaged over the whole batch ``mean_edit_distance``

    and retain the original ``labels`` (Y).
    ``decoded``, ``labels``, the optimization gradient, ``distance``, ``mean_edit_distance``,
    ``total_loss`` and ``avg_loss`` are collected into the corresponding arrays
    ``tower_decodings``, ``tower_labels``, ``tower_gradients``, ``tower_distances``,
    ``tower_mean_edit_distances``, ``tower_total_losses``, ``tower_avg_losses`` (dimension 0 being the tower).
    Finally this new method ``get_tower_results()`` will return those tower arrays.
    In case of ``tower_mean_edit_distances`` and ``tower_avg_losses``, it will return the
    averaged values instead of the arrays.
    '''
    # Tower labels to return
    tower_labels = []

    # Tower decodings to return
    tower_decodings = []

    # Tower distances to return
    tower_distances = []

    # Tower total batch losses to return
    tower_total_losses = []

    # Tower gradients to return
    tower_gradients = []

    # To calculate the mean of the mean edit distances
    tower_mean_edit_distances = []

    # To calculate the mean of the losses
    tower_avg_losses = []

    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(available_devices)):
            # Execute operations of tower i on device i
            if len(FLAGS.ps_hosts) == 0:
                device = available_devices[i]
            else:
                device = tf.train.replica_device_setter(worker_device=available_devices[i], cluster=cluster)
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i) as scope:
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    total_loss, avg_loss, distance, mean_edit_distance, decoded, labels = \
                        calculate_mean_edit_distance_and_loss(model_feeder, i, training=training)
                        #calculate_mean_edit_distance_and_loss(model_feeder, i, training = (optimizer is None))


                    # Allow for variables to be re-used by the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Retain tower's labels (Y)
                    tower_labels.append(labels)

                    # Retain tower's decoded batch
                    tower_decodings.append(decoded)

                    # Retain tower's distances
                    tower_distances.append(distance)

                    # Retain tower's total losses
                    tower_total_losses.append(total_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # mask inf and nans in gradient
                    gradients = [(mask_nans(gv[0]), gv[1]) for gv in gradients ]

                    # Retain tower's gradients
                    tower_gradients.append(gradients)

                    # Retain tower's mean edit distance
                    tower_mean_edit_distances.append(mean_edit_distance)

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

    # Return the results tuple, the gradients, and the means of mean edit distances and losses
    return (tower_labels, tower_decodings, tower_distances, tower_total_losses), \
           tower_gradients, \
           tf.reduce_mean(tower_mean_edit_distances, 0), \
           tf.reduce_mean(tower_avg_losses, 0)


def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a syncronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(cpu_device):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []

            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads



# Logging
# =======

def log_variable(variable, gradient=None):
    r'''
    We introduce a function for logging a tensor variable's current state.
    It logs scalar values for the mean, standard deviation, minimum and maximum.
    Furthermore it logs a histogram of its state and (if given) of an optimization gradient.
    '''
    name = variable.name
    mean = tf.reduce_mean(variable)
    tf.summary.scalar(name='%s/mean'   % name, tensor=mean)
    tf.summary.scalar(name='%s/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(tf.square(variable - mean))))
    tf.summary.scalar(name='%s/max'    % name, tensor=tf.reduce_max(variable))
    tf.summary.scalar(name='%s/min'    % name, tensor=tf.reduce_min(variable))
    tf.summary.histogram(name=name, values=variable)
    if gradient is not None:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is not None:
            mean = tf.reduce_mean(grad_values)
            tf.summary.scalar(name='%s/gradients/mean' % name, tensor=mean)
            tf.summary.scalar(name='%s/gradients/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(tf.square(grad_values - mean))))
            tf.summary.scalar(name='%s/gradients/max' % name, tensor=tf.reduce_max(grad_values))
            tf.summary.scalar(name='%s/gradients/min' % name, tensor=tf.reduce_min(grad_values))

            tf.summary.histogram(name='%s/gradients' % name, values=grad_values)


def log_grads_and_vars(grads_and_vars):
    r'''
    Let's also introduce a helper function for logging collections of gradient/variable tuples.
    '''
    for gradient, variable in grads_and_vars:
        log_variable(variable, gradient=gradient)

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

def get_git_branch():
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()


# Helpers
# =======

def calculate_report(results_tuple):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = []
    items = list(zip(*results_tuple))
    total_levenshtein = 0.0
    total_label_length = 0.0
    for label, decoding, distance, loss in items:
        sample_wer = wer(label, decoding)
        sample = Sample(label, decoding, loss, distance, sample_wer)
        samples.append(sample)
        total_levenshtein += levenshtein(label.split(), decoding.split())
        total_label_length += float(len(label.split()))

    # Getting the mean WER from the accumulated levenshteins and lengths
    samples_wer = total_levenshtein / total_label_length

    # Filter out all items with WER=0
    samples = [s for s in samples if s.wer > 0]

    # Order the remaining items by their loss (lowest loss on top)
    samples.sort(key=lambda s: s.loss)

    # Take only the first report_count items
    samples = samples[:FLAGS.report_count]

    # Order this top FLAGS.report_count items by their WER (lowest WER on top)
    samples.sort(key=lambda s: s.wer)

    return samples_wer, samples

def collect_results(results_tuple, returns):
    r'''
    This routine will help collecting partial results for the WER reports.
    The ``results_tuple`` is composed of an array of the original labels,
    an array of the corresponding decodings, an array of the corrsponding
    distances and an array of the corresponding losses. ``returns`` is built up
    in a similar way, containing just the unprocessed results of one
    ``session.run`` call (effectively of one batch).
    Labels and decodings are converted to text before splicing them into their
    corresponding results_tuple lists. In the case of decodings,
    for now we just pick the first available path.
    '''
    # Each of the arrays within results_tuple will get extended by a batch of each available device
    for i in range(len(available_devices)):
        # Collect the labels
        results_tuple[0].extend(sparse_tensor_value_to_texts(returns[0][i], alphabet))

        # Collect the decodings - at the moment we default to the first one
        results_tuple[1].extend(sparse_tensor_value_to_texts(returns[1][i][0], alphabet))

        # Collect the distances
        results_tuple[2].extend(returns[2][i])

        # Collect the losses
        results_tuple[3].extend(returns[3][i])


# For reporting we also need a standard way to do time measurements.
def stopwatch(start_duration=0):
    r'''
    This function will toggle a stopwatch.
    The first call starts it, second call stops it, third call continues it etc.
    So if you want to measure the accumulated time spent in a certain area of the code,
    you can surround that code by stopwatch-calls like this:

    .. code:: python

        fun_time = 0 # initializes a stopwatch
        [...]
        for i in range(10):
          [...]
          # Starts/continues the stopwatch - fun_time is now a point in time (again)
          fun_time = stopwatch(fun_time)
          fun()
          # Pauses the stopwatch - fun_time is now a duration
          fun_time = stopwatch(fun_time)
        [...]
        # The following line only makes sense after an even call of :code:`fun_time = stopwatch(fun_time)`.
        print 'Time spent in fun():', format_duration(fun_time)

    '''
    if start_duration == 0:
        return time.time()
    else:
        return time.time() - start_duration

def format_duration(duration):
    '''Formats the result of an even stopwatch call as hours:minutes:seconds:milliseconds'''
    # duration = duration if isinstance(duration, int) else duration.seconds
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%.3f' % (h, m, s)


# Execution
# =========

# String constants for different services of the web handler
PREFIX_NEXT_INDEX = '/next_index_'
PREFIX_GET_JOB = '/get_job_'

# Global ID counter for all objects requiring an ID
id_counter = 0

def new_id():
    '''Returns a new ID that is unique on process level. Not thread-safe.

    Returns:
        int. The new ID
    '''
    global id_counter
    id_counter += 1
    return id_counter

class Sample(object):
    def __init__(self, src, res, loss, mean_edit_distance, sample_wer):
        '''Represents one item of a WER report.

        Args:
            src (str): source text
            res (str): resulting text
            loss (float): computed loss of this item
            mean_edit_distance (float): computed mean edit distance of this item
        '''
        self.src = src
        self.res = res
        self.loss = loss
        self.mean_edit_distance = mean_edit_distance
        self.wer = sample_wer

    def __str__(self):
        return 'WER: %f, loss: %f, mean edit distance: %f\n - src: "%s"\n - res: "%s"' % (self.wer, self.loss, self.mean_edit_distance, self.src, self.res)

class WorkerJob(object):
    def __init__(self, epoch_id, index, set_name, steps, report):
        '''Represents a job that should be executed by a worker.

        Args:
            epoch_id (int): the ID of the 'parent' epoch
            index (int): the epoch index of the 'parent' epoch
            set_name (str): the name of the data-set - one of 'train', 'dev', 'test'
            steps (int): the number of `session.run` calls
            report (bool): if this job should produce a WER report
        '''
        self.id = new_id()
        self.epoch_id = epoch_id
        self.index = index
        self.worker = -1
        self.set_name = set_name
        self.steps = steps
        self.report = report
        self.loss = -1
        self.mean_edit_distance = -1
        self.wer = -1
        self.samples = []

    def __str__(self):
        return 'Job (ID: %d, worker: %d, epoch: %d, set_name: %s)' % (self.id, self.worker, self.index, self.set_name)

class Epoch(object):
    '''Represents an epoch that should be executed by the Training Coordinator.
    Creates `num_jobs` `WorkerJob` instances in state 'open'.

    Args:
        index (int): the epoch index of the 'parent' epoch
        num_jobs (int): the number of jobs in this epoch

    Kwargs:
        set_name (str): the name of the data-set - one of 'train', 'dev', 'test'
        report (bool): if this job should produce a WER report
    '''
    def __init__(self, index, num_jobs, set_name='train', report=False):
        self.id = new_id()
        self.index = index
        self.num_jobs = num_jobs
        self.set_name = set_name
        self.report = report
        self.wer = -1
        self.loss = -1
        self.mean_edit_distance = -1
        self.jobs_open = []
        self.jobs_running = []
        self.jobs_done = []
        self.samples = []
        for i in range(self.num_jobs):
            self.jobs_open.append(WorkerJob(self.id, self.index, self.set_name, FLAGS.iters_per_worker, self.report))

    def name(self):
        '''Gets a printable name for this epoch.

        Returns:
            str. printable name for this epoch
        '''
        if self.index >= 0:
            ename = ' of Epoch %d' % self.index
        else:
            ename = ''
        if self.set_name == 'train':
            return 'Training%s' % ename
        elif self.set_name == 'dev':
            return 'Validation%s' % ename
        else:
            return 'Test%s' % ename

    def get_job(self, worker):
        '''Gets the next open job from this epoch. The job will be marked as 'running'.

        Args:
            worker (int): index of the worker that takes the job

        Returns:
            WorkerJob. job that has been marked as running for this worker
        '''
        if len(self.jobs_open) > 0:
            job = self.jobs_open.pop(0)
            self.jobs_running.append(job)
            job.worker = worker
            return job
        else:
            return None

    def finish_job(self, job):
        '''Finishes a running job. Removes it from the running jobs list and adds it to the done jobs list.

        Args:
            job (WorkerJob): the job to put into state 'done'
        '''
        index = next((i for i in range(len(self.jobs_running)) if self.jobs_running[i].id == job.id), -1)
        if index >= 0:
            self.jobs_running.pop(index)
            self.jobs_done.append(job)
            log_traffic('%s - Moved %s from running to done.' % (self.name(), job))
        else:
            log_warn('%s - There is no job with ID %d registered as running.' % (self.name(), job.id))

    def done(self):
        '''Checks, if all jobs of the epoch are in state 'done'.
        It also lazy-prepares a WER report from the result data of all jobs.

        Returns:
            bool. if all jobs of the epoch are 'done'
        '''
        if len(self.jobs_open) == 0 and len(self.jobs_running) == 0:
            num_jobs = len(self.jobs_done)
            if num_jobs > 0:
                jobs = self.jobs_done
                self.jobs_done = []
                if not self.num_jobs == num_jobs:
                    log_warn('%s - Number of steps not equal to number of jobs done.' % (self.name()))

                agg_loss = 0.0
                agg_wer = 0.0
                agg_mean_edit_distance = 0.0

                for i in range(num_jobs):
                    job = jobs.pop(0)
                    agg_loss += job.loss
                    if self.report:
                        agg_wer += job.wer
                        agg_mean_edit_distance += job.mean_edit_distance
                        self.samples.extend(job.samples)

                self.loss = agg_loss / num_jobs

                # if the job was for validation dataset then append it to the COORD's _loss for early stop verification
                if (FLAGS.early_stop is True) and (self.set_name == 'dev'):
                    COORD._dev_losses.append(self.loss)

                if self.report:
                    self.wer = agg_wer / num_jobs
                    self.mean_edit_distance = agg_mean_edit_distance / num_jobs

                    # Order samles by their loss (lowest loss on top)
                    self.samples.sort(key=lambda s: s.loss)

                    # Take only the first report_count items
                    self.samples = self.samples[:FLAGS.report_count]

                    # Order this top FLAGS.report_count items by their WER (lowest WER on top)
                    self.samples.sort(key=lambda s: s.wer)

                    # Append WER to WER log file
                    if len(FLAGS.wer_log_pattern) > 0:
                        time = datetime.datetime.utcnow().isoformat()
                        # Log WER progress
                        print(FLAGS.wer_log_pattern % (time, self.set_name, self.wer))

            return True
        return False

    def job_status(self):
        '''Provides a printable overview of the states of the jobs of this epoch.

        Returns:
            str. printable overall job state
        '''
        return '%s - jobs open: %d, jobs running: %d, jobs done: %d' % (self.name(), len(self.jobs_open), len(self.jobs_running), len(self.jobs_done))

    def __str__(self):
        if not self.done():
            return self.job_status()

        if not self.report:
            return '%s - loss: %f' % (self.name(), self.loss)

        s = '%s - WER: %f, loss: %s, mean edit distance: %f' % (self.name(), self.wer, self.loss, self.mean_edit_distance)
        if len(self.samples) > 0:
            line = '\n' + ('-' * 80)
            for sample in self.samples:
                s += '%s\n%s' % (line, sample)
            s += line
        return s


class TrainingCoordinator(object):
    class TrainingCoordinationHandler(BaseHTTPServer.BaseHTTPRequestHandler):
        '''Handles HTTP requests from remote workers to the Training Coordinator.
        '''
        def _send_answer(self, data=None):
            self.send_response(200)
            self.send_header('content-type', 'text/plain')
            self.end_headers()
            if data:
                self.wfile.write(data)

        def do_GET(self):
            if COORD.started:
                if self.path.startswith(PREFIX_NEXT_INDEX):
                    index = COORD.get_next_index(self.path[len(PREFIX_NEXT_INDEX):])
                    if index >= 0:
                        self._send_answer(str(index).encode("utf-8"))
                        return
                elif self.path.startswith(PREFIX_GET_JOB):
                    job = COORD.get_job(worker=int(self.path[len(PREFIX_GET_JOB):]))
                    if job:
                        self._send_answer(pickle.dumps(job))
                        return
                self.send_response(204) # end of training
            else:
                self.send_response(202) # not ready yet
            self.end_headers()

        def do_POST(self):
            if COORD.started:
                src = self.rfile.read(int(self.headers['content-length']))
                job = COORD.next_job(pickle.loads(src))
                if job:
                    self._send_answer(pickle.dumps(job))
                    return
                self.send_response(204) # end of training
            else:
                self.send_response(202) # not ready yet
            self.end_headers()

        def log_message(self, format, *args):
            '''Overriding base method to suppress web handler messages on stdout.
            '''
            return


    def __init__(self):
        ''' Central training coordination class.
        Used for distributing jobs among workers of a cluster.
        Instantiated on all workers, calls of non-chief workers will transparently
        HTTP-forwarded to the chief worker instance.
        '''
        self._init()
        self._lock = Lock()
        self.started = False
        if is_chief:
            self._httpd = BaseHTTPServer.HTTPServer((FLAGS.coord_host, FLAGS.coord_port), TrainingCoordinator.TrainingCoordinationHandler)

    def _reset_counters(self):
        self._index_train = 0
        self._index_dev = 0
        self._index_test = 0

    def _init(self):
        self._epochs_running = []
        self._epochs_done = []
        self._reset_counters()
        self._dev_losses = []

    def _log_all_jobs(self):
        '''Use this to debug-print epoch state'''
        log_debug('Epochs - running: %d, done: %d' % (len(self._epochs_running), len(self._epochs_done)))
        for epoch in self._epochs_running:
            log_debug('       - running: ' + epoch.job_status())

    def start_coordination(self, model_feeder, step=0):
        '''Starts to coordinate epochs and jobs among workers on base of
        data-set sizes, the (global) step and FLAGS parameters.

        Args:
            model_feeder (ModelFeeder): data-sets to be used for coordinated training

        Kwargs:
            step (int): global step of a loaded model to determine starting point
        '''
        with self._lock:
            self._init()

            # Number of GPUs per worker - fixed for now by local reality or cluster setup
            gpus_per_worker = len(available_devices)

            # Number of batches processed per job per worker
            batches_per_job  = gpus_per_worker * max(1, FLAGS.iters_per_worker)

            # Number of batches per global step
            batches_per_step = gpus_per_worker * max(1, FLAGS.replicas_to_agg)

            # Number of global steps per epoch - to be at least 1
            steps_per_epoch = max(1, model_feeder.train.total_batches // batches_per_step)

            # The start epoch of our training
            self._epoch = step // steps_per_epoch

            # Number of additional 'jobs' trained already 'on top of' our start epoch
            jobs_trained = (step % steps_per_epoch) * batches_per_step // batches_per_job

            # Total number of train/dev/test jobs covering their respective whole sets (one epoch)
            self._num_jobs_train = max(1, model_feeder.train.total_batches // batches_per_job)
            self._num_jobs_dev   = max(1, model_feeder.dev.total_batches   // batches_per_job)
            self._num_jobs_test  = max(1, model_feeder.test.total_batches  // batches_per_job)

            if FLAGS.epoch < 0:
                # A negative epoch means to add its absolute number to the epochs already computed
                self._target_epoch = self._epoch + abs(FLAGS.epoch)
            else:
                self._target_epoch = FLAGS.epoch

            # State variables
            # We only have to train, if we are told so and are not at the target epoch yet
            self._train = FLAGS.train and self._target_epoch > self._epoch
            self._test = FLAGS.test

            if self._train:
                # The total number of jobs for all additional epochs to be trained
                # Will be decremented for each job that is produced/put into state 'open'
                self._num_jobs_train_left = (self._target_epoch - self._epoch) * self._num_jobs_train - jobs_trained
                log_info('STARTING Optimization')
                self._training_time = stopwatch()

            # Important for debugging
            log_debug('step: %d' % step)
            log_debug('epoch: %d' % self._epoch)
            log_debug('target epoch: %d' % self._target_epoch)
            log_debug('steps per epoch: %d' % steps_per_epoch)
            log_debug('number of batches in train set: %d' % model_feeder.train.total_batches)
            log_debug('batches per job: %d' % batches_per_job)
            log_debug('batches per step: %d' % batches_per_step)
            log_debug('number of jobs in train set: %d' % self._num_jobs_train)
            log_debug('number of jobs already trained in first epoch: %d' % jobs_trained)

            self._next_epoch()

        # The coordinator is ready to serve
        self.started = True

    def _next_epoch(self):
        # State-machine of the coodination process

        # Indicates, if there were 'new' epoch(s) provided
        result = False

        # Make sure that early stop is enabled and validation part is enabled
        if (FLAGS.early_stop is True) and (FLAGS.validation_step > 0) and (len(self._dev_losses) >= FLAGS.earlystop_nsteps):

            # Calculate the mean of losses for past epochs
            mean_loss = np.mean(self._dev_losses[-FLAGS.earlystop_nsteps:-1])
            # Calculate the standard deviation for losses from validation part in the past epochs
            std_loss = np.std(self._dev_losses[-FLAGS.earlystop_nsteps:-1])
            # Update the list of losses incurred
            self._dev_losses = self._dev_losses[-FLAGS.earlystop_nsteps:]
            log_debug('Checking for early stopping (last %d steps) validation loss: %f, with standard deviation: %f and mean: %f'
                      % (FLAGS.earlystop_nsteps, self._dev_losses[-1], std_loss, mean_loss))

            # Check if validation loss has started increasing or is not decreasing substantially, making sure slight fluctuations don't bother the early stopping from working
            if self._dev_losses[-1] > np.max(self._dev_losses[:-1]) or (abs(self._dev_losses[-1] - mean_loss) < FLAGS.estop_mean_thresh
                                                                        and std_loss < FLAGS.estop_std_thresh):
                # Time to early stop
                log_info('Early stop triggered as (for last %d steps) validation loss: %f with standard deviation: %f and mean: %f'
                         % (FLAGS.earlystop_nsteps, self._dev_losses[-1], std_loss, mean_loss))
                self._dev_losses = []
                self._end_training()
                self._train = False

        if self._train:
            # We are in train mode
            if self._num_jobs_train_left > 0:
                # There are still jobs left
                num_jobs_train = min(self._num_jobs_train_left, self._num_jobs_train)
                self._num_jobs_train_left -= num_jobs_train

                # Let's try our best to keep the notion of curriculum learning
                self._reset_counters()

                # If the training part of the current epoch should generate a WER report
                is_display_step = FLAGS.display_step > 0 and (FLAGS.display_step == 1 or self._epoch > 0) and (self._epoch % FLAGS.display_step == 0 or self._epoch == self._target_epoch)
                # Append the training epoch
                self._epochs_running.append(Epoch(self._epoch, num_jobs_train, set_name='train', report=is_display_step))

                if FLAGS.validation_step > 0 and (FLAGS.validation_step == 1 or self._epoch > 0) and self._epoch % FLAGS.validation_step == 0:
                    # The current epoch should also have a validation part
                    self._epochs_running.append(Epoch(self._epoch, self._num_jobs_dev, set_name='dev', report=is_display_step))


# ADD save checkpoint




                # Indicating that there were 'new' epoch(s) provided
                result = True
            else:
                # No jobs left, but still in train mode: concluding training
                self._end_training()
                self._train = False

        if self._test and not self._train:
            # We shall test, and are not in train mode anymore
            self._test = False
            self._epochs_running.append(Epoch(self._epoch, self._num_jobs_test, set_name='test', report=True))
            # Indicating that there were 'new' epoch(s) provided
            result = True

        if result:
            # Increment the epoch index - shared among train and test 'state'
            self._epoch += 1
        return result

    def _end_training(self):
        self._training_time = stopwatch(self._training_time)
        log_info('FINISHED Optimization - training time: %s' % format_duration(self._training_time))

    def start(self):
        '''Starts Training Coordinator. If chief, it starts a web server for
        communication with non-chief instances.
        '''
        if is_chief:
            log_debug('Starting coordinator...')
            self._thread = Thread(target=self._httpd.serve_forever)
            self._thread.daemon = True
            self._thread.start()
            log_debug('Coordinator started.')

    def stop(self, wait_for_running_epochs=True):
        '''Stops Training Coordinator. If chief, it waits for all epochs to be
        'done' and then shuts down the web server.
        '''
        if is_chief:
            if wait_for_running_epochs:
                while len(self._epochs_running) > 0:
                    log_traffic('Coordinator is waiting for epochs to finish...')
                    time.sleep(5)
            log_debug('Stopping coordinator...')
            self._httpd.shutdown()
            log_debug('Coordinator stopped.')

    def _talk_to_chief(self, path, data=None, default=None):
        tries = 0
        while tries < FLAGS.coord_retries:
            tries += 1
            try:
                url = 'http://%s:%d%s' % (FLAGS.coord_host, FLAGS.coord_port, path)
                log_traffic('Contacting coordinator - url: %s, tries: %d ...' % (url, tries-1))
                res = urllib.request.urlopen(urllib.request.Request(url, data, { 'content-type': 'text/plain' }))
                str = res.read()
                status = res.getcode()
                log_traffic('Coordinator responded - url: %s, status: %s' % (url, status))
                if status == 200:
                    return str
                if status == 204: # We use 204 (no content) to indicate end of training
                    return default
            except urllib.error.HTTPError as error:
                log_traffic('Problem reaching coordinator - url: %s, HTTP code: %d' % (url, error.code))
                pass
            time.sleep(10)
        return default

    def get_next_index(self, set_name):
        '''Retrives a new cluster-unique batch index for a given set-name.
        Prevents applying one batch multiple times per epoch.

        Args:
            set_name (str): name of the data set - one of 'train', 'dev', 'test'

        Returns:
            int. new data set index
        '''
        with self._lock:
            if is_chief:
                member = '_index_' + set_name
                value = getattr(self, member, -1)
                setattr(self, member, value + 1)
                return value
            else:
                # We are a remote worker and have to hand over to the chief worker by HTTP
                log_traffic('Asking for next index...')
                value = int(self._talk_to_chief(PREFIX_NEXT_INDEX + set_name))
                log_traffic('Got index %d.' % value)
                return value

    def _get_job(self, worker=0):
        job = None
        # Find first running epoch that provides a next job
        for epoch in self._epochs_running:
            job = epoch.get_job(worker)
            if job:
                return job
        # No next job found
        return None

    def get_job(self, worker=0):
        '''Retrieves the first job for a worker.

        Kwargs:
            worker (int): index of the worker to get the first job for

        Returns:
            WorkerJob. a job of one of the running epochs that will get
                associated with the given worker and put into state 'running'
        '''
        # Let's ensure that this does not interfer with other workers/requests
        with self._lock:
            if is_chief:
                # First try to get a next job
                job = self._get_job(worker)

                if job is None:
                    # If there was no next job, we give it a second chance by triggering the epoch state machine
                    if self._next_epoch():
                        # Epoch state machine got a new epoch
                        # Second try to get a next job
                        job = self._get_job(worker)
                        if job is None:
                            # Albeit the epoch state machine got a new epoch, the epoch had no new job for us
                            log_error('Unexpected case - no job for worker %d.' % (worker))
                        return job

                    # Epoch state machine has no new epoch
                    # This happens at the end of the whole training - nothing to worry about
                    log_traffic('No jobs left for worker %d.' % (worker))
                    self._log_all_jobs()
                    return None

                # We got a new job from one of the currently running epochs
                log_traffic('Got new %s' % job)
                return job

            # We are a remote worker and have to hand over to the chief worker by HTTP
            result = self._talk_to_chief(PREFIX_GET_JOB + str(FLAGS.task_index))
            if result:
                result = pickle.loads(result)
            return result

    def next_job(self, job):
        '''Sends a finished job back to the coordinator and retrieves in exchange the next one.

        Kwargs:
            job (WorkerJob): job that was finished by a worker and who's results are to be
                digested by the coordinator

        Returns:
            WorkerJob. next job of one of the running epochs that will get
                associated with the worker from the finished job and put into state 'running'
        '''
        if is_chief:
            # Try to find the epoch the job belongs to
            epoch = next((epoch for epoch in self._epochs_running if epoch.id == job.epoch_id), None)
            if epoch:
                # We are going to manipulate things - let's avoid undefined state
                with self._lock:
                    # Let the epoch finish the job
                    epoch.finish_job(job)
                    # Check, if epoch is done now
                    if epoch.done():
                        # If it declares itself done, move it from 'running' to 'done' collection
                        self._epochs_running.remove(epoch)
                        self._epochs_done.append(epoch)
                        log_info('%s' % epoch)
            else:
                # There was no running epoch found for this job - this should never happen.
                log_error('There is no running epoch of ID %d for job with ID %d. This should never happen.' % (job.epoch_id, job.id))
            return self.get_job(job.worker)

        # We are a remote worker and have to hand over to the chief worker by HTTP
        result = self._talk_to_chief('', data=pickle.dumps(job))
        if result:
            result = pickle.loads(result)
        return result

def send_token_to_ps(session, kill=False):
    # Sending our token (the task_index as a debug opportunity) to each parameter server.
    # kill switch tokens are negative and decremented by 1 to deal with task_index 0
    token = -FLAGS.task_index-1 if kill else FLAGS.task_index
    kind = 'kill switch' if kill else 'stop'
    for index, enqueue in enumerate(done_enqueues):
        log_debug('Sending %s token to ps %d...' % (kind, index))
        session.run(enqueue, feed_dict={ token_placeholder: token })
        log_debug('Sent %s token to ps %d.' % (kind, index))

def train(server=None):
    r'''
    Trains the network on a given server of a cluster.
    If no server provided, it performs single process training.
    '''

    # Create a variable to hold the global_step.
    # It will automgically get incremented by the optimizer.
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Reading training set
    train_set = DataSet(FLAGS.train_files.split(','),
                        FLAGS.train_batch_size,
                        limit=FLAGS.limit_train,
                        ascending=FLAGS.train_sort,
                        next_index=lambda i: COORD.get_next_index('train'),
                        shuffle=True,
                        augmentation=augmentation_parameters if FLAGS.augment else None)

    # Reading validation set
    dev_set = DataSet(FLAGS.dev_files.split(','),
                      FLAGS.dev_batch_size,
                      limit=FLAGS.limit_dev,
                      next_index=lambda i: COORD.get_next_index('dev'),
                      shuffle=False,
                      augmentation=None)

    # Reading test set
    test_set = DataSet(FLAGS.test_files.split(','),
                       FLAGS.test_batch_size,
                       limit=FLAGS.limit_test,
                       next_index=lambda i: COORD.get_next_index('test'),
                       shuffle=False,
                       augmentation=None)

    # Combining all sets to a multi set model feeder
    model_feeder = ModelFeeder(train_set,
                               dev_set,
                               test_set,
                               n_input,
                               n_context,
                               alphabet,
                               tower_feeder_count=len(available_devices),
                               input_type=input_type,
                               reduction_factor=reduction_factor,
                               numpad=n_pad
)

    if (FLAGS.decay_steps > 0) and (FLAGS.decay_rate > 0):
        lr = tf.train.exponential_decay(learning_rate = FLAGS.learning_rate,
                                             global_step   = global_step,
                                             decay_steps   = FLAGS.decay_steps,
                                             decay_rate    = FLAGS.decay_rate,
                                             staircase = True)
    else:
        lr = tf.convert_to_tensor(FLAGS.learning_rate)

    # Create the optimizer
    optimizer = create_optimizer(optimizer = FLAGS.optimizer,lr=lr)

    # Synchronous distributed training is facilitated by a special proxy-optimizer
    if not server is None:
        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                   replicas_to_aggregate=FLAGS.replicas_to_agg,
                                                   total_num_replicas=FLAGS.replicas)

    # Get the data_set specific graph end-points
    results_tuple, gradients, mean_edit_distance, loss = get_tower_results(model_feeder, optimizer, training= True)

    # Average tower gradients across GPUs
    avg_tower_gradients = average_gradients(gradients)

    # Add summaries of all variables and gradients to log
    log_grads_and_vars(avg_tower_gradients)

    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.scalar(name='mean_edit_distance', tensor=mean_edit_distance)

    # Op to merge all summaries for the summary hook
    merge_all_summaries_op = tf.summary.merge_all()

    # Apply gradients to modify the model
    apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)


    if FLAGS.early_stop is True and not FLAGS.validation_step > 0:
        log_warn('Parameter --validation_step needs to be >0 for early stopping to work')

    class CoordHook(tf.train.SessionRunHook):
        r'''
        Embedded coordination hook-class that will use variables of the
        surrounding Python context.
        '''
        def after_create_session(self, session, coord):
            log_debug('Starting queue runners...')
            model_feeder.start_queue_threads(session, coord)
            log_debug('Queue runners started.')

        def end(self, session):
            # Closing the data_set queues
            log_debug('Closing queues...')
            model_feeder.close_queues(session)
            log_debug('Queues closed.')

            # Telling the ps that we are done
            send_token_to_ps(session)

    # Collecting the hooks
    hooks = [CoordHook()]

    # Hook to handle initialization and queues for sync replicas.
    if not server is None:
        hooks.append(optimizer.make_session_run_hook(is_chief))

    # Hook to save TensorBoard summaries
    if (FLAGS.summary_secs is not None) or (FLAGS.summary_steps is not None):
        hooks.append(tf.train.SummarySaverHook(save_secs=FLAGS.summary_secs, save_steps=FLAGS.summary_steps,
                                           output_dir=FLAGS.summary_dir, summary_op=merge_all_summaries_op))

    # Hook wih number of checkpoint files to save in checkpoint_dir
    if FLAGS.train and FLAGS.max_to_keep > 0:
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.checkpoint_dir, save_secs=FLAGS.checkpoint_secs, saver=saver))

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    try:
        with tf.train.MonitoredTrainingSession(master='' if server is None else server.target,
                                               is_chief=is_chief,
                                               hooks=hooks,
                                               checkpoint_dir=FLAGS.checkpoint_dir,
                                               save_checkpoint_secs=FLAGS.checkpoint_secs if FLAGS.train else None,
                                               config=session_config) as session:
            try:
                if is_chief:
                    # Retrieving global_step from the (potentially restored) model
                    feed_dict = {}
                    model_feeder.set_data_set(feed_dict, model_feeder.train)
                    step = session.run(global_step, feed_dict=feed_dict)
                    COORD.start_coordination(model_feeder, step)

                session_time = 0.0

                # Get the first job
                job = COORD.get_job()
                while job and not session.should_stop():
                    log_debug('Computing %s...' % job)

                    # The feed_dict (mainly for switching between queues)
                    feed_dict = {}

                    # Sets the current data_set for the respective placeholder in feed_dict
                    model_feeder.set_data_set(feed_dict, getattr(model_feeder, job.set_name))
                    # Initialize loss aggregator
                    total_loss = 0.0

                    # Setting the training operation in case of training requested
                    train_op = apply_gradient_op if job.set_name == 'train' else []

                    # Requirements to display a WER report
                    if job.report:
                        # Reset mean edit distance
                        total_mean_edit_distance = 0.0
                        # Create report results tuple
                        report_results = ([],[],[],[])
                        # Extend the session.run parameters
                        report_params = [results_tuple, mean_edit_distance]
                    else:
                        report_params = []

                    # So far the only extra parameter is the feed_dict
                    extra_params = { 'feed_dict': feed_dict}

                    # Loop over the batches
                    for job_step in range(job.steps):
                        if session.should_stop():
                            break

                        log_debug('Starting batch...')
                        # Compute the batch
                        batch_time = time.time()
#                        _, current_step, batch_loss, batch_report = session.run([train_op, global_step, loss, report_params], **extra_params)
                        _, current_step, batch_loss, learn_rate, batch_report = session.run([train_op,
                                                                global_step, loss, lr, report_params], **extra_params)


                        batch_time = time.time() - batch_time
                        session_time += batch_time
                        # Uncomment the next line for debugging race conditions / distributed TF
                        log_debug('Finished batch step %d %f' %(current_step, batch_loss))
                        if ((current_step % 100) == 0):
                            log_info('time: %s, step: %d, loss: %f lr: %f' %
                                      (format_duration(session_time), current_step, batch_loss, learn_rate)
                                     )
                            session_time = 0.0

                        # Add batch to loss
                        total_loss += batch_loss

                        if job.report:
                            # Collect individual sample results
                            collect_results(report_results, batch_report[0])
                            # Add batch to total_mean_edit_distance
                            total_mean_edit_distance += batch_report[1]

                    # Gathering job results
                    job.loss = total_loss / job.steps
                    if job.report:
                        job.mean_edit_distance = total_mean_edit_distance / job.steps
                        job.wer, job.samples = calculate_report(report_results)


                    # Send the current job to coordinator and receive the next one
                    log_debug('Sending %s...' % job)
                    job = COORD.next_job(job)
            except Exception as e:
                log_error(str(e))
                traceback.print_exc()
                # Calling all hook's end() methods to end blocking calls
                for hook in hooks:
                    hook.end(session)
                # Only chief has a SyncReplicasOptimizer queue runner that needs to be stopped for unblocking process exit.
                # A rather graceful way to do this is by stopping the ps.
                # Only one party can send it w/o failing.
                if is_chief:
                    send_token_to_ps(session, kill=True)
                sys.exit(1)

        log_debug('Session closed.')

    except tf.errors.InvalidArgumentError as e:
        log_error(str(e))
        log_error('The checkpoint in {0} does not match the shapes of the model.'
                  ' Did you change alphabet.txt or the --n_hidden parameter'
                  ' between train runs using the same checkpoint dir? Try moving'
                  ' or removing the contents of {0}.'.format(FLAGS.checkpoint_dir))
        sys.exit(1)

def create_inference_graph(batch_size=None, output_is_logits=False, use_new_decoder=False):
    # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
    input_tensor = tf.placeholder(tf.float32, [batch_size, None, n_input + 2*n_input*n_context], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    # TODO : fix training  from bool to tensor
    logits = DeepSpeech2(input_tensor, tf.to_int64(seq_length) if FLAGS.use_seq_length else None, training=None)

    if output_is_logits:
        return logits

    # Beam search decode the batch
    decoder = decode_with_lm if use_new_decoder else tf.nn.ctc_beam_search_decoder

    decoded, _ = decoder(logits, seq_length, merge_repeated=False, beam_width=FLAGS.beam_width)
    decoded = tf.convert_to_tensor(
        [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded], name='output_node')

    return (
        {
            'input': input_tensor,
            'input_lengths': seq_length,
        },
        {
            'outputs': decoded,
        }
    )


def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
    with tf.device('/cpu:0'):

        tf.reset_default_graph()
        session = tf.Session(config=session_config)

        inputs, outputs = create_inference_graph()

        # TODO: Transform the decoded output to a string

        # Create a saver and exporter using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())
        model_exporter = exporter.Exporter(saver)

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counterract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)
        log_info('Restored checkpoint at training epoch %d' % (int(checkpoint_path.split('-')[-1]) + 1))

        # Initialise the model exporter and export the model
        model_exporter.init(session.graph.as_graph_def(),
                            named_graph_signatures = {
                                'inputs': exporter.generic_signature(inputs),
                                'outputs': exporter.generic_signature(outputs)
                            })
        if FLAGS.remove_export:
            actual_export_dir = os.path.join(FLAGS.export_dir, '%08d' % FLAGS.export_version)
            if os.path.isdir(actual_export_dir):
                log_info('Removing old export')
                shutil.rmtree(actual_FLAGS.export_dir)
        try:
            # Export serving model
            model_exporter.export(FLAGS.export_dir, tf.constant(FLAGS.export_version), session)

            # Export graph
            input_graph_name = 'input_graph.pb'
            tf.train.write_graph(session.graph, FLAGS.export_dir, input_graph_name, as_text=False)

            # Freeze graph
            input_graph_path = os.path.join(FLAGS.export_dir, input_graph_name)
            input_saver_def_path = ''
            input_binary = True
            output_node_names = 'output_node'
            restore_op_name = 'save/restore_all'
            filename_tensor_name = 'save/Const:0'
            output_graph_path = os.path.join(FLAGS.export_dir, 'output_graph.pb')
            clear_devices = False
            freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                      input_binary, checkpoint_path, output_node_names,
                                      restore_op_name, filename_tensor_name,
                                      output_graph_path, clear_devices, '')

            log_info('Models exported at %s' % (FLAGS.export_dir))
        except RuntimeError:
            log_error(sys.exc_info()[1])


def do_single_file_inference(input_file_path):
    with tf.Session(config=session_config) as session:
        inputs, outputs = create_inference_graph(batch_size=1, use_new_decoder=True)

        # Create a saver using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counterract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        mfcc = audiofile_to_input_vector(input_file_path, n_input, n_context, input_type)
        output = session.run(outputs['outputs'], feed_dict = {
            inputs['input']: [mfcc],
            inputs['input_lengths']: [len(mfcc)],
        })

        text = ndarray_to_text(output[0][0], alphabet)

        print(text)


def main(_) :

    initialize_globals()

    if FLAGS.train or FLAGS.test:
        if len(FLAGS.worker_hosts) == 0:
            # Only one local task: this process (default case - no cluster)
            train()
            log_debug('Done.')
        else:
            # Create and start a server for the local task.
            server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
            if FLAGS.job_name == 'ps':
                # We are a parameter server and therefore we just wait for all workers to finish
                # by waiting for their stop tokens.
                with tf.Session(server.target) as session:
                    for worker in FLAGS.worker_hosts:
                        log_debug('Waiting for stop token...')
                        token = session.run(done_dequeues[FLAGS.task_index])
                        if token < 0:
                            log_debug('Got a kill switch token from worker %i.' % abs(token + 1))
                            break
                        log_debug('Got a stop token from worker %i.' % token)
                log_debug('Session closed.')
            elif FLAGS.job_name == 'worker':
                # We are a worker and therefore we have to do some work.
                # Assigns ops to the local worker by default.
                with tf.device(tf.train.replica_device_setter(
                               worker_device=worker_device,
                               cluster=cluster)):

                    # Do the training
                    train(server)

            log_debug('Server stopped.')

    # Are we the main process?
    if is_chief:
        # Doing solo/post-processing work just on the main process...
        # Exporting the model
        if FLAGS.export_dir:
            export()

    if len(FLAGS.one_shot_infer):
        do_single_file_inference(FLAGS.one_shot_infer)

    # Stopping the coordinator
    COORD.stop()

if __name__ == '__main__' :
    tf.app.run()
