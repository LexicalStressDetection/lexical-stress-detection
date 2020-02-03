# coding=utf-8
# acoustic features include:
# New Features
# 5. duration, normalized duration
# 6. pitch max, pitch min, pitch median
# 7. zero corssings number (Not Used), zero crossings ratio (Not Used)
# 8. phone_before_vowel, phone_after_vowel, phoneme
# 9. energy, energy entropy (Not Used)
# 10.RMS (Not Used)
# 11.Spectral Fulx (Not Used), Spectral RollOff (Not Used)


import pandas as pd
import numpy
import math
import wave
import scipy.io.wavfile as wav

from scipy.fftpack import dct
from scipy.signal import lfilter, fftconvolve
import time


eps = 1e-8 # 0.00000001

def get_duration(phone):
    """
    This function returns the duration of phoneme audio
    """
    len_frames = phone.getnframes()
    rate = phone.getframerate()
    return len_frames / float(rate)


def duration_norm(duration, waveFile):
    """
    This function returns normalized duration: duration(phone) / duration(word)
    """
    audiofile = wave.open(waveFile, "r")
    params = audiofile.getparams()
    rate = params[2]
    frames = params[3]
    waveDuration = frames / float(rate)
    normDuration = duration / float(waveDuration)
    return normDuration


def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Refer to feature_extraction.py
    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def pitch_from_zcr(frame, fs):
    """
    The function detects the F0 of isolated phoneme by zero-crossing
    """
    M = numpy.round(0.016 * fs) - 1
    #print (frames.shape)
    R = numpy.correlate(frame, frame, mode='full')
    g = R[len(frame)-1]
    R = R[len(frame):-1]
    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))
    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]

    if M > len(R):
        M = len(R) - 1

    M = int(M)
    m0 = int(m0)
    Gamma = numpy.zeros(M)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g * CSum[M:m0:-1])) + eps)
    ZCR = zcr(Gamma)
    if ZCR[1] > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)
        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0
    pitch = f0
    return HR, pitch

def get_window_rms(frame, window_size = 2):
    """
    The function returns root mean square value of each given frame
    """
    frame = numpy.power(frame, 2)
    window = numpy.ones(window_size) / float(window_size)
    return numpy.sqrt(numpy.convolve(frame, window, 'valid'))

# Rising and Falling time frames
def zcr(frame):
    """
    Compute the number and rate of sign-changes of the signal during the duration of a particular frame
    """
    count = len(frame)
    countZC = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return countZC, (numpy.float64(countZC) / numpy.float64(count - 1.0))

def getEnergy(frame):
    """
    Compute energy value of frame
    """
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))

def energyEntropy(frame, block = 10):
    """
    Compute entropy of sub-fames’ normalized energies
    """
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    win_len = int(numpy.floor(L / block))
    if L != win_len * block:
            frame = frame[0 : win_len * block]
    sub_wins = frame.reshape(win_len, block, order = 'F').copy()
    sub_wins_entropy = numpy.sum(sub_wins ** 2, axis = 0) / (Eol + eps) # normalized sub-frame energies
    Entropy = -numpy.sum(sub_wins_entropy * numpy.log2(sub_wins_entropy + eps)) # entropy of the normalized sub-frame energies
    return Entropy


def SpectralFlux(X, prevX):
    """
    Compute the squared difference between the normalized magnitudes of the spectra of the two successive frames
    """
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(prevX + eps)
    flux = numpy.sum(( X / sumX - prevX / sumPrevX) ** 2)
    return flux


def SpectralRollOff(X, c, fs):
    """
    Compute the frequency below which 90% of the magnitude distribution of the spectrum is concentrated
    """
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    thres = c * totalEnergy
    cumsum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(cumsum > thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return mC


