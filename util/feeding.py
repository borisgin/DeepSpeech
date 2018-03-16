import pandas
import tensorflow as tf
import numpy as np

from threading import Thread
from math import ceil
from six.moves import range
from util.audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.text import ctc_label_dense_to_sparse, text_to_char_array

class ModelFeeder(object):
    '''
    Feeds data into a model.
    Feeding is parallelized by independent units called tower feeders (usually one per GPU).
    Each tower feeder provides data from three runtime switchable sources (train, dev, test).
    These sources are to be provided by three DataSet instances whos references are kept.
    Creates, owns and delegates to tower_feeder_count internal tower feeder objects.
    '''
    def __init__(self,
                 train_set,
                 dev_set,
                 test_set,
                 numcep,
                 numcontext,
                 alphabet,
                 tower_feeder_count=-1,
                 threads_per_queue=2,
                 input_type='mfcc',
                 reduction_factor=1,
                 numpad=0,
                 dummy_seq_len=0,
                 dummy_num_iters=0
        ):

        self.train = train_set
        self.dev = dev_set
        self.test = test_set
        self.dummy_seq_len = dummy_seq_len
        self.dummy_num_iters = dummy_num_iters
        self.mode = 0
        self.sets = [train_set, dev_set, test_set]
        self.numcep = numcep
        self.numcontext = numcontext
        self.input_type = input_type
        self.reduction_factor  = reduction_factor
        self.numpad = numpad
        self.tower_feeder_count = max(len(get_available_gpus()), 1) if tower_feeder_count < 0 else tower_feeder_count
        self.threads_per_queue = threads_per_queue

        self.ph_x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self.ph_x_length = tf.placeholder(tf.int32, [])
        self.ph_y = tf.placeholder(tf.int32, [None,])
        self.ph_y_length = tf.placeholder(tf.int32, [])

        self.ph_batch_size = tf.placeholder(tf.int32, [])
        self.ph_queue_selector = tf.placeholder(tf.int32, name='Queue_Selector')

        self._tower_feeders = [_TowerFeeder(self, i, alphabet) for i in range(self.tower_feeder_count)]

    def start_queue_threads(self, session, coord):
        '''
        Starts required queue threads on all tower feeders.
        '''
        queue_threads = []
        for tower_feeder in self._tower_feeders:
            queue_threads += tower_feeder.start_queue_threads(session, coord)
        return queue_threads

    def close_queues(self, session):
        '''
        Closes queues of all tower feeders.
        '''
        for tower_feeder in self._tower_feeders:
            tower_feeder.close_queues(session)

    def set_data_set(self, feed_dict, data_set):
        '''
        Switches all tower feeders to a different source DataSet.
        The provided feed_dict will get enriched with required placeholder/value pairs.
        The DataSet has to be one of those that got passed into the constructor.
        '''
        index = self.sets.index(data_set)
        assert index >= 0
        self.mode = index

        feed_dict[self.ph_queue_selector] = index
        feed_dict[self.ph_batch_size] = data_set.batch_size


    def next_batch(self, tower_feeder_index):
        '''
        Draw the next batch from one of the tower feeders.
        '''
        return self._tower_feeders[tower_feeder_index].next_batch()

class DataSet(object):
    '''
    Represents a collection of audio samples and their respective transcriptions.
    Takes a set of CSV files produced by importers in /bin.
    '''
    def __init__(self, csvs, batch_size, skip=0, limit=0, ascending=False, next_index=lambda i: i + 1, shuffle=True,
                 augmentation=None, dummy_seq_len=0, dummy_num_iters=0):
        self.batch_size = batch_size
        self.next_index = next_index
        self.dummy_seq_len = dummy_seq_len
        self.dummy_num_iters = dummy_num_iters
        self.shuffle = shuffle
        if dummy_num_iters == 0:
            self.files = None
            self.augmentation = augmentation
            for csv in csvs:
                file = pandas.read_csv(csv, encoding='utf-8')
                if self.files is None:
                    self.files = file
                else:
                    self.files = self.files.append(file)
            if ascending:
                self.files = self.files.sort_values(by="wav_filesize", ascending=ascending) \
                         .loc[:, ["wav_filename", "transcript"]] \
                         .values[skip:]
            else:
                self.files = self.files.loc[:, ["wav_filename", "transcript"]] \
                             .values[skip:]
            if limit > 0:
                self.files = self.files[:limit]
            if shuffle:
                self.files = np.random.permutation(self.files)
        else:
            self.files = np.array(['dummy'] * (dummy_num_iters * batch_size))
        self.total_batches = int(ceil(len(self.files) / batch_size))

class _DataSetLoader(object):
    '''
    Internal class that represents an input queue with data from one of the DataSet objects.
    Each tower feeder will create and combine three data set loaders to one switchable queue.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    Keeps a DataSet reference to access its samples.
    '''
    def __init__(self, model_feeder, data_set, alphabet):
        self._model_feeder = model_feeder
        self._data_set = data_set
        self.queue = tf.PaddingFIFOQueue(shapes=[[None, model_feeder.numcep + (2 * model_feeder.numcep * model_feeder.numcontext)], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=data_set.batch_size * 2)

        self._enqueue_op = self.queue.enqueue([model_feeder.ph_x, model_feeder.ph_x_length,
                                               model_feeder.ph_y, model_feeder.ph_y_length  ])
        self._close_op = self.queue.close(cancel_pending_enqueues=True)
        self._alphabet = alphabet

    def start_queue_threads(self, session, coord):
        '''
        Starts concurrent queue threads for reading samples from the data set.
        '''
        queue_threads = [Thread(target=self._populate_batch_queue, args=(session, coord))
                         for i in range(self._model_feeder.threads_per_queue)]
        for queue_thread in queue_threads:
            coord.register_thread(queue_thread)
            queue_thread.daemon = True
            queue_thread.start()
        return queue_threads

    def close_queue(self, session):
        '''
        Closes the data set queue.
        '''
        session.run(self._close_op)

    def _populate_batch_queue(self, session, coord):
        '''
        Queue thread routine.
        '''
        file_count = len(self._data_set.files)
        index = -1
        while not coord.should_stop():
            index = self._data_set.next_index(index)
            # Reshuffle dataset after every epoch
            if index >= file_count:
                if self._data_set.shuffle:
                    self._data_set.files = np.random.permutation(self._data_set.files)
                index = 0
            if self._data_set.dummy_seq_len == 0:
                wav_file, transcript = self._data_set.files[index]
                target = text_to_char_array(transcript, self._alphabet)
                target_len = len(target)

                source = audiofile_to_input_vector(wav_file, self._model_feeder.numcep, self._model_feeder.numcontext,
                                                   self._model_feeder.input_type,
                                                   augmentation=self._data_set.augmentation)

                source_len = len(source)

                # TODO: move fix ctc
                min_len = target_len * self._model_feeder.reduction_factor
                if source_len < min_len:
                    numpad = (min_len - source_len) // 2
                    print('char_len={} audio_len={} pad={}'.format(target_len, len(source), numpad))
                    num_features=self._model_feeder.numcep
                    pad_shape=[numpad, num_features]
                    start_pad = np.broadcast_to(source[0, :], pad_shape)
                    end_pad   = np.broadcast_to(source[-1,:], pad_shape)
                    source = np.concatenate((start_pad, source, end_pad))
                    # pad = np.zeros([numpad, self._model_feeder.numcep])
                    # source = np.concatenate((pad, source, pad))

                #--------------------------------------------------

                num_features = self._model_feeder.numcep
                numpad= self._model_feeder.numpad
                pad_shape = [numpad, num_features]
                start_pad = np.broadcast_to(source[0, :], pad_shape)
                end_pad = np.broadcast_to(source[-1, :], pad_shape)
                source = np.concatenate((start_pad, source, end_pad))

                #if source_len // self._model_feeder.reduction_factor < target_len:
                #    print("audio {}, chars {}".format(source.shape, target_len))
                #    raise ValueError('Error: Audio file {} is too short for transcription.'.format(wav_file))

                #print(source.shape, len(source))
            else:
                source_len = self._data_set.dummy_seq_len
                source = np.random.randn(source_len, self._model_feeder.numcep).astype(np.float32)
                target_len = int(self._data_set.dummy_seq_len/4)
                target = np.ones(shape=target_len)
               
            try:
                session.run(self._enqueue_op, feed_dict={ self._model_feeder.ph_x: source,
                                                          self._model_feeder.ph_x_length: len(source),
                                                          self._model_feeder.ph_y: target,
                                                          self._model_feeder.ph_y_length: target_len
                                                          })
            except tf.errors.CancelledError:
                return

class _TowerFeeder(object):
    '''
    Internal class that represents a switchable input queue for one tower.
    It creates, owns and combines three _DataSetLoader instances.
    Keeps a ModelFeeder reference for accessing shared settings and placeholders.
    '''
    def __init__(self, model_feeder, index, alphabet):
        self._model_feeder = model_feeder
        self.index = index
        self._loaders = [_DataSetLoader(model_feeder, data_set, alphabet) for data_set in model_feeder.sets]
        self._queues = [set_queue.queue for set_queue in self._loaders]
        self._queue = tf.QueueBase.from_list(model_feeder.ph_queue_selector, self._queues)
        self._close_op = self._queue.close(cancel_pending_enqueues=True)

    def next_batch(self):
        '''
        Draw the next batch from from the combined switchable queue.
        '''
        source, source_lengths, target, target_lengths = self._queue.dequeue_many(self._model_feeder.ph_batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._model_feeder.ph_batch_size)
        return source, source_lengths, sparse_labels, self._model_feeder.ph_queue_selector

    def start_queue_threads(self, session, coord):
        '''
        Starts the queue threads of all owned _DataSetLoader instances.
        '''
        queue_threads = []
        for set_queue in self._loaders:
            queue_threads += set_queue.start_queue_threads(session, coord)
        return queue_threads

    def close_queues(self, session):
        '''
        Closes queues of all owned _DataSetLoader instances.
        '''
        for set_queue in self._loaders:
            set_queue.close_queue(session)

