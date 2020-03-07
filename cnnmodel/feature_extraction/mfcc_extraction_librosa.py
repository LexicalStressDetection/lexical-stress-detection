import numpy as np
import librosa


def get_mfcc(signal, samplerate):
    # in librosa the window length and step size (stride) are set by number of frames and not
    # duration. window_length is set by n_fft and step is set by hop_length
    frame_length = int(0.025 * samplerate)
    step_size = int(0.01 * samplerate)
    mfcc = librosa.feature.mfcc(signal, samplerate, n_mfcc=13, n_fft=frame_length, hop_length=step_size)
    mfcc_derivative = librosa.feature.delta(mfcc, order=1)
    mfcc_second_derivative = librosa.feature.delta(mfcc, order=2)

    assert mfcc.shape == (13, 10)
    assert mfcc_derivative.shape == (13, 10)
    assert mfcc_second_derivative.shape == (13, 10)

    # stack mfcc, derivative and second derivative horizontally
    mfcc_matrix = np.concatenate([mfcc, mfcc_derivative, mfcc_second_derivative], axis=1)
    assert mfcc_matrix.shape == (13, 30)

    return mfcc_matrix
