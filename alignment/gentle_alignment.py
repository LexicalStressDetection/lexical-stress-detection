import json
import logging
import multiprocessing
import os

import gentle
import scipy.io.wavfile as sciwav

DISFLUENCIES = {'uh', 'um'}  # set of disfluencies
RESOURCES = gentle.Resources()
N_THREADS = multiprocessing.cpu_count()

logging.getLogger().setLevel("INFO")


def _on_progress(p):
    for k, v in p.items():
        logging.debug("%s: %s" % (k, v))


def _get_key_val_pair(line):
    line_split = line[:-1].split()
    word = line_split[0]
    if word[-1] == ')':
        word = word.split('(')[0]

    word = word.lower()
    key = [word]
    val = []
    for phoneme in line_split[1:]:
        val.append(phoneme.lower())
        if phoneme[-1].isdigit():
            phoneme = phoneme[:-1]

        phoneme = phoneme.lower()
        key.append(phoneme)

    key = " ".join(key)
    val = tuple(val)
    return key, val


def _create_dict():
    phoneme_alignment_dict = dict()

    cmu_file = open('cmudict-0.7b.txt', 'r')
    for line in cmu_file:
        key, val = _get_key_val_pair(line)
        phoneme_alignment_dict[key] = val

    return phoneme_alignment_dict


def align_audio(wav_path, transcript):
    with gentle.resampled(wav_path) as wavfile:
        print("starting alignment {}".format(wav_path))
        aligner = gentle.ForcedAligner(RESOURCES, transcript, nthreads=N_THREADS, disfluency=False,
                                       conservative=False, disfluencies=DISFLUENCIES)
        result = aligner.transcribe(wavfile, progress_cb=_on_progress, logging=logging)
        result_json = json.loads(result.to_json())

    return result_json


def main(input_csv, phoneme_path, output_csv, wav_root):
    alignment_dict = _create_dict()

    in_file = open(input_csv, 'r')
    out_file = open(output_csv, 'w')

    for line in in_file:
        id_, wav_file, transcript = line.split('\t')
        wav_file = wav_root + '/' + wav_file
        sr, signal = sciwav.read(wav_file)
        alignment = align_audio(wav_file, transcript)

        for word in alignment['words']:
            if word['case'] != 'success':
                continue

            start_time, end_time = word['start'], word['end']
            aligned_word = word['alignedWord']
            key = [aligned_word.lower()]
            for phoneme in word['phones']:
                phone = phoneme['phone']
                key.append(phone.split('_')[0])

            key = ' '.join(key)
            phoneme_tuple = alignment_dict.get(key, ())

            if len(phoneme_tuple) == 0:
                print('word: {} not in dict, skipping...'.format(word))
                continue

            if len(phoneme_tuple) != len(word['phones']):
                print('word: {} not aligned properly, skipping...'.format(word))
                continue

            # now map phonemes and slice wav
            for i, phoneme in enumerate(word['phones']):
                phone_start = start_time
                phone_end = phone_start + phoneme['duration']
                # check if vowel phoneme
                if phoneme_tuple[i][-1].isdigit():

                    file_name = id_ + '_' + aligned_word + '_' + phoneme_tuple[i] + '_' + \
                                str(int(phone_start * 1000)) + '_' + str(int(phone_end * 1000)) + '.wav'

                    start_frame, end_frame = int(phone_start * sr), int(phone_end * sr)
                    sciwav.write(phoneme_path + '/' + file_name, sr, signal[start_frame:end_frame])
                    out_file.write(file_name + '\t' + id_ + '\t' + aligned_word + '\t' + phoneme_tuple[i] + '\n')

                start_time = phone_end

        print('done alignment and slicing for file: {}'.format(wav_file))

    in_file.close()
    out_file.close()


if __name__ == '__main__':
    main(input_csv=os.getenv('input_csv'),
         phoneme_path=os.getenv('phoneme_path'),
         output_csv=os.getenv('output_csv'),
         wav_root=os.getenv('wav_root'))
