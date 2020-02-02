import numpy
import feature_extraction
import feature_extraction_stress


def p2pamplitude(signal):
    """
    f1 : Compute the peak-to-peak amplitude of the signal
    """
    return numpy.max(signal) - numpy.min(signal)

def get_energy_for_frame(frame):
    """
    Compute energy value of frame
    """
    return feature_extraction_stress.getEnergy(frame)

def get_energy_for_frames(frames):
    """
    Compute energy value for all frames
    """
    energy = []
    for i in range(len(frames)):
        energy.append(get_energy_for_frame(frames[i]))
    return energy


def mean_energy_over_syllable_nucleus(energy):
    """
    f2 : Mean energy over syllable nucleus
    """
    return numpy.mean(energy)

def max_energy_over_syllable_nucleus(energy):
    """
    f3 : Max energy over syllable nucleus
    """
    return numpy.max(energy)

def get_duration(sound_wave):
    """
    f4 & f5 : This function returns the duration of a sound wave. Send input (syllable/vowel) accordingly
    """
    return feature_extraction_stress.get_duration(sound_wave)

def get_pitch(signal, fs):
    return feature_extraction.freq_from_autocorr(signal, fs)

def get_MFCC(signal, fs, cep_num=27):
    return feature_extraction.calcMFCC(signal, fs, cep_num)







