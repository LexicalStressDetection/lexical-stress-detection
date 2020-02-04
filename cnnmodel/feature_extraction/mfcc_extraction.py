import numpy
import time
import math
from scipy.fftpack import dct


def audio2frame(signal, frame_length, frame_step, winfunc=lambda x: numpy.ones((x,))):
    """
    Frame a signal into overlapping frames.
    :param signal: the audio signal to frame.
    :param frame_length: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    if signal_length <= frame_length:
        frames_num = 1
    else:
        frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))

    pad_length = int((frames_num - 1) * frame_step + frame_length)

    zeros = numpy.zeros((pad_length - signal_length,))
    pad_signal = numpy.concatenate((signal, zeros))

    indices = numpy.tile(numpy.arange(0, frame_length), (frames_num, 1)) + numpy.tile(
        numpy.arange(0, frames_num * frame_step, frame_step), (frame_length, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    frames = pad_signal[indices]
    win = numpy.tile(winfunc(frame_length), (frames_num, 1))

    return frames * win


def pre_emphasis(signal, coefficient=0.95):
    """
    perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    if signal[0] != [] and signal[1:] != [] and signal[:-1] != []:
        return numpy.append(signal[0], signal[1:] - coefficient * signal[:-1])
    else:
        time.sleep(0.7)
        return numpy.append(signal[0], signal[1:] - coefficient * signal[:-1])


def fbank(signal, samplerate=16000, win_length=0.025, win_step=0.01,
          filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    """
    Compute Mel-filterbank energy features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param win_length: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param win_step: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    The second return value is the energy in each frame (total energy, unwindowed)
    """
    high_freq = high_freq or samplerate / 2
    signal = pre_emphasis(signal, pre_emphasis_coeff)
    frames = audio2frame(signal, win_length * samplerate, win_step * samplerate)
    spec_power = spectrum_power(frames, NFFT)
    energy = numpy.sum(spec_power, 1)
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)

    fb = get_filter_banks(filters_num, NFFT, samplerate, low_freq, high_freq)
    feat = numpy.dot(spec_power, fb.T)
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)
    return feat, energy


def calc_MFCC(signal, samplerate=16000, win_length=0.025, win_step=0.01,
              cep_num=13, filters_num=26, NFFT=512, low_freq=0, high_freq=None,
              pre_emphasis_coeff=0.97, cep_lifter=22, appendEnergy=True):
    """
    Compute MFCC features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param win_length: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param win_step: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param cep_num: the number of cepstrum to return, default 13
    :param filters_num: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowf_req: lowest band edge of mel filters. In Hz, default is 0.
    :param high_freq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param pre_emphasis_coeff: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param cep_lifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, win_length, win_step, filters_num,
                         NFFT, low_freq, high_freq, pre_emphasis_coeff)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :cep_num]
    feat = lifter(feat, cep_lifter)
    if appendEnergy: feat[:, 0] = numpy.log(energy)
    return feat


def spectrum_power(frames, NFFT):
    return 1.0 / NFFT * numpy.square(spectrum_magnitude(frames, NFFT))


def spectrum_magnitude(frames, NFFT):
    complex_spectrum = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spectrum)


def hz2mel(hz):
    """
    Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.0)


def mel2hz(mel):
    """
    Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filter_banks(filters_num=20, NFFT=512, samplerate=16000, low_freq=0, high_freq=None):
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    mel_points = numpy.linspace(low_mel, high_mel, filters_num + 2)
    hz_points = mel2hz(mel_points)
    bin = numpy.floor((NFFT + 1) * hz_points / samplerate)
    fbank = numpy.zeros([filters_num, NFFT // 2 + 1])
    for j in range(0, filters_num):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def lifter(cepstra, L=22):
    """
    Apply a cepstral lifter the the matrix of cepstra.
    This has the effect of increasing the magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L / 2) * numpy.sin(numpy.pi * n / L)
        return lift * cepstra
    else:
        return cepstra


def get_mfcc(signal, samplerate, cep_num=27):
    """
    27 Mel-scale energy bands over syllable nucleus
    """
    return calc_MFCC(signal, samplerate, cep_num)
