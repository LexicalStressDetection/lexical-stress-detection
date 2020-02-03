import numpy
import feature_extraction
import feature_extraction_stress


def get_p2pamplitude(signal):
    """
    f1 : Compute the peak-to-peak amplitude of the signal
    """
    return numpy.max(signal) - numpy.min(signal)


def get_mean_energy_over_syllable_nucleus(energy):
    """
    f2 : Mean energy over syllable nucleus
    """
    return numpy.mean(energy)


def get_max_energy_over_syllable_nucleus(energy):
    """
    f3 : Max energy over syllable nucleus
    """
    return numpy.max(energy)


def get_duration(sound_wave):
    """
    f4 & f5 : Duration of a sound wave. Send input (syllable/vowel) accordingly
    """
    return feature_extraction_stress.get_duration(sound_wave)


def get_max_pitch_over_syllable_nucleus(pitch_for_frames):
    """
    f6 : Maximum pitch over syllable nucleus
    """
    return numpy.max(pitch_for_frames)


def get_mean_pitch_over_syllable_nucleus(pitch_for_frames):
    """
    f7 : Mean pitch over syllable nucleus
    """
    return numpy.mean(pitch_for_frames)


def get_mfcc(signal, fs, cep_num=27):
    """
    27 Mel-scale energy bands over syllable nucleus
    """
    return feature_extraction.calcMFCC(signal, fs, cep_num)


def pitch_from_zcr(frame, fs):
    """
    Compute pitch for each frame
    """
    return feature_extraction_stress.pitch_from_zcr(frame, fs)[1]


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


def get_pitch_values(frames, fs):
    """
    Compute pitch values for all frames
    """
    pitch_for_frames = []
    for i in range(len(frames)):
        pitch_for_frames.append(pitch_from_zcr(frames[i], fs))
    return pitch_for_frames
