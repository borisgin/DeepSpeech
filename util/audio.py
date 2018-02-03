from __future__ import absolute_import, print_function

import scipy.io.wavfile as wav
import sys

try:
    from deepspeech.utils import audioToInputVector_DO_NOT_IMPORT
except ImportError:
    import numpy as np
    from python_speech_features import mfcc
    from python_speech_features.sigproc import framesig, logpowspec, powspec
    from six.moves import range

    import resampy as rs

    class DeprecationWarning:
        displayed = False

    def audioToInputVector(audio, fs, numcep, numcontext, input_type='mfcc', augmentation=None):
        if DeprecationWarning.displayed is not True:
            DeprecationWarning.displayed = True
            print('------------------------------------------------------------------------', file=sys.stderr)
            print('WARNING: using python_speech_features: type {} num_featues {} '.format(input_type, numcep),
                  file=sys.stderr)
            print('------------------------------------------------------------------------', file=sys.stderr)

        if augmentation is not None:
            # data augmentation
            audio_float = audio.astype(np.float32)/32768.0

            if augmentation['time_stretch_ratio'] > 0:
                # time stretch (might be slow)
                stretch_amount = 1.0 + (2*np.random.rand()-1) * augmentation['time_stretch_ratio']
                audio_float = rs.resample(audio_float, fs, int(fs/stretch_amount), filter='kaiser_fast')

            # noise
            noise_level_db = np.random.randint(low=augmentation['noise_level_min'],
                                               high=augmentation['noise_level_max'])
            audio_float += np.random.randn(len(audio_float))*10**(noise_level_db/20.0)

            audio = (audio_float*32768.0).astype(np.int16)

        if input_type == 'spectrogram':
            assert numcep % 2 == 1, "m_input shouldn't be even for spectrogram"
            frames = framesig(sig=audio,
                              frame_len=int(fs*0.020),
                              frame_step=int(fs*0.010),
                              winfunc=np.hanning)

            # TODO: try log(1+powspec)
            # train_inputs = np.log1p(powspec(frames, NFFT=(numcep-1)*2))
            train_inputs = logpowspec(frames, NFFT=(numcep-1)*2)
        elif input_type == 'mfcc':
            # Get mfcc coefficients
            #features = mfcc(audio, samplerate=fs, numcep=numcep)

            features = mfcc(audio, samplerate=fs, winlen=0.025, winstep=0.01,
                 numcep=numcep,
                 nfilt= 2*numcep,
                 nfft=512,
                 lowfreq=0, highfreq=None,
                 preemph=0.97,
                 ceplifter= 2*numcep,  #22,
                 appendEnergy=False)

            # We only keep every second feature (BiRNN stride = 2)
            #features = features[::2]

            # One stride per time step in the input
            num_strides = len(features)

            # Add empty initial and final contexts
            empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
            features = np.concatenate((empty_context, features, empty_context))

            # Create a view into the array with overlapping strides of size
            # numcontext (past) + 1 (present) + numcontext (future)
            window_size = 2*numcontext+1
            train_inputs = np.lib.stride_tricks.as_strided(
                features,
                (num_strides, window_size, numcep),
                (features.strides[0], features.strides[0], features.strides[1]),
                writeable=False)

            # Flatten the second and third dimensions
            train_inputs = np.reshape(train_inputs, [num_strides, -1])
            # Copy the strided array so that we can write to it safely
            train_inputs = np.copy(train_inputs)
        else:
            raise ValueError('Unknown input type: {}'.format(input_type))

        # Whiten inputs (TODO: Should we whiten?)
        m = np.mean(train_inputs)
        v = np.std(train_inputs)
        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - m) / v
        # Return results
        return train_inputs


def audiofile_to_input_vector(audio_filename, numcep, numcontext, input_type='mfcc',
                              augmentation=None):
    r"""
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    # Load wav files
    fs, audio = wav.read(audio_filename)

    return audioToInputVector(audio, fs, numcep, numcontext, input_type, augmentation)
