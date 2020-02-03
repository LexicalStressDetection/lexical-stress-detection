# coding=utf-8
# acoustic features include:
# 1. 13 MFCC + 13 first-order differential coefficients + 13 acceleration coefficients a total of 39 coefficients
# 2. duration
# 3. pitch (F0)
# 4. formants (F1, F2, F3, F4)

import pandas as pd
import numpy
import math
import wave
import contextlib
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from scipy.signal import lfilter, fftconvolve
import time


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


def deframesignal(frames, signal_length, frame_length, frame_step, winfunc=lambda x: numpy.ones((x,))):
    """
	overlap-add procedure to undo the action of framesig.
	:param frames: the array of frames.
	:param siglen_length: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
	:param frame_length: length of each frame measured in samples.
	:param frame_step: number of samples after the start of the previous frame that the next frame should begin.
	:param winfunc: the analysis window to apply to each frame. By default no window is applied.
	:returns: a 1-D signal.
	"""
    signal_length = round(signal_length)
    frame_length = round(frame_length)
    frames_num = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_length, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
    indices = numpy.tile(numpy.arange(0, frame_length), (frames_num, 1)) + numpy.tile(
        numpy.arange(0, frames_num * frame_step, frame_step), (frame_length, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    pad_length = (frames_num - 1) * frame_step + frame_length
    if signal_length <= 0:
        signal_length = pad_length
    recalc_signal = numpy.zeros((pad_length,))
    window_correction = numpy.zeros((pad_length, 1))
    win = winfunc(frame_length)
    for i in range(0, frames_num):
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + win + 1e-15
        recalc_signal[indices[i, :]] = recalc_signal[indices[i, :]] + frames[i, :]
    recalc_signal = recalc_signal / window_correction
    return recalc_signal[0:signal_length]


def spectrum_magnitude(frames, NFFT):
    complex_spectrum = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spectrum)


def spectrum_power(frames, NFFT):
    return 1.0 / NFFT * numpy.square(spectrum_magnitude(frames, NFFT))


def log_spectrum_power(frames, NFFT, norm=1):
    """
    Compute the log power spectrum of each frame in frames.
    If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    spec_power = spectrum_power(frames, NFFT)
    spec_power[spec_power < 1e-30] = 1e-30
    log_spec_power = 10 * numpy.log10(spec_power)
    if norm:
        return log_spec_power - numpy.max(log_spec_power)
    else:
        return log_spec_power


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


def calcMFCC_delta_delta(signal, samplerate=16000, win_length=0.025, win_step=0.01,
                         cep_num=13, filters_num=26, NFFT=512, low_freq=0, high_freq=None,
                         pre_emphasis_coeff=0.97, cep_lifter=22, appendEnergy=True):
    '''
    Calculate 13 MFCC + 13 first-order differential coefficients + 13 acceleration coefficients
    a total of 39 coefficients
    '''
    feat = calcMFCC(signal, samplerate, win_length, win_step, cep_num,
                    filters_num, NFFT, low_freq, high_freq, pre_emphasis_coeff, cep_lifter, appendEnergy)
    result1 = derivate(feat)
    result2 = derivate(result1)
    result3 = numpy.concatenate((feat, result1), axis=1)
    result = numpy.concatenate((result3, result2), axis=1)
    return result


def derivate(feat, big_theta=2, cep_num=13):
    '''
    General transformation formula for calculating first-order coefficients or acceleration coefficients
    '''
    result = numpy.zeros(feat.shape)
    denominator = 0
    for theta in numpy.linspace(1, big_theta, big_theta):
        denominator = denominator + theta ** 2
    denominator = denominator * 2
    for row in numpy.linspace(0, feat.shape[0] - 1, feat.shape[0]):
        tmp = numpy.zeros((cep_num,))
        numerator = numpy.zeros((cep_num,))
        for t in numpy.linspace(1, cep_num, cep_num):
            a = 0
            b = 0
            s = 0
            for theta in numpy.linspace(1, big_theta, big_theta):
                if (t + theta) > cep_num:
                    a = 0
                else:
                    a = feat[int(row)][int(t + theta - 1)]
                if (t - theta) < 1:
                    b = 0
                else:
                    b = feat[int(row)][int(t - theta - 1)]
                s += theta * (a - b)
            numerator[int(t - 1)] = s
        tmp = numerator * 1.0 / denominator
        result[int(row)] = tmp
    return result


def calcMFCC(signal, samplerate=16000, win_length=0.025, win_step=0.01,
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


def log_fbank(signal, samplerate=16000, win_length=0.025, win_step=0.01,
              filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    feat, energy = fbank(signal, samplerate, win_length, win_step, filters_num, NFFT,
                         low_freq, high_freq, pre_emphasis_coeff)
    return numpy.log(feat)


def ssc(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26,
        NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    """
    Compute Spectral Subband Centroid features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param win_length: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param win_step: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param filters_num: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param low_freq: lowest band edge of mel filters. In Hz, default is 0.
    :param high_freq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param pre_emphasis_coeff: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    high_freq = high_freq or samplerate / 2
    signal = pre_emphasis(signal, pre_emphasis_coeff)
    frames = audio2frame(signal, win_length * samplerate, win_step * samplerate)
    spec_power = spectrum_power(frames, NFFT)
    spec_power = numpy.where(spec_power == 0, numpy.finfo(float).eps, spec_power)
    fb = get_filter_banks(filters_num, NFFT, samplerate, low_freq, high_freq)
    feat = numpy.dot(spec_power, fb.T)
    R = numpy.tile(numpy.linspace(1, samplerate / 2, numpy.size(spec_power, 1)), (numpy.size(spec_power, 0), 1))
    return numpy.dot(spec_power * R, fb.T) / feat


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


def lpc(y, m):
    "Return m linear predictive coefficients for sequence y using Levinson-Durbin prediction algorithm"
    #step 1: compute autoregression coefficients R_0, ..., R_m
    R = [y.dot(y)]
    if R[0] == 0:
        return [1] + [0] * (m-2) + [-1]
    else:
        for i in range(1, m + 1):
            r = y[i:].dot(y[:-i])
            R.append(r)
        R = numpy.array(R)
    #step 2:
        A = numpy.array([1, -R[1] / R[0]])
        E = R[0] + R[1] * A[1]
        for k in range(1, m):
            if (E == 0):
                E = 10e-17
            alpha = - A[:k+1].dot(R[k+1:0:-1]) / E
            A = numpy.hstack([A,0])
            A = A + alpha * A[::-1]
            E *= (1 - alpha**2)
        return A


def formants(signal, rate):
    """
    Formants(F1,F2,F3,F4) estimation using LPC by adapting the following matlab code:
    http://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html
    """
    # Hamming window
    N = len(signal)
    w = numpy.hamming(N)
    # Apply window and high pass filter
    x1 = signal * w
    # x1 = lfilter([1., 0.63], 1, x1)
    x1 = lfilter([1], [1., 0.63], x1)
    # LPC
    ncoeff = 2 + rate // \
             1000
    A = lpc(x1, ncoeff)
    rts = numpy.roots(A)
    rts = [r for r in rts if numpy.imag(r) >= 0]
    angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))
    # Get frequencies.
    frqs = sorted(angz * (rate / (2 * math.pi)))
    return frqs


def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental.
    It's also best method for our task to detect the F0 of isolated phoneme,
    more accuracy than zero-crossing (long data length) and
    peak of FFT (also work well for longer data length)
    Cons: Not as accurate, currently has trouble with finding the true peak
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr) // 2:]    
    # Find the first low point
    d = numpy.diff(corr)
    start_tmp = [i for i, e in enumerate(d) if e > 0]
    if start_tmp != []:
        start = start_tmp[0]  # find(d > 0)[0]
        # Find the next peak after the low point (other than 0 lag).  This bit is
        # not reliable, due to peaks that occur between samples.
        peak = numpy.argmax(corr[start:]) + start
        if numpy.int(peak + 1) != int(len(corr)):
            px, py = parabolic(corr, peak)
            return fs / px
    # start = [i for i, e in enumerate(d) if e >0][0] #find(d > 0)[0]
    # # Find the next peak after the low point (other than 0 lag).  This bit is
    # # not reliable, due to peaks that occur between samples.
    # peak = numpy.argmax(corr[start:]) + start
    # px, py = parabolic(corr, peak)
    # return fs / px


def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1 / 2 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4 * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)
