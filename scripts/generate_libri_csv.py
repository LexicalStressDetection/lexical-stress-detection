import os
import sys


def main(libri_root, out_file):
    out_file = open(out_file, 'w')
    for top_dir in os.listdir(libri_root):
        if top_dir == 'train-clean-100' or top_dir == 'train-clean-360':
            for speaker in os.listdir(libri_root + '/' + top_dir):
                for section in os.listdir(libri_root + '/' + top_dir + '/' + speaker):
                    trans_file = libri_root + '/' + top_dir + '/' + speaker + '/' + section + '/' + \
                                 speaker + '-' + section + '.trans.txt'

                    with open(trans_file, 'r') as t:
                        for line in t:
                            id_, transcript = line[:-1].split(' ', 1)
                            transcript = transcript.lower()
                            audio_file_path = top_dir + '/' + speaker + '/' + section + '/' + \
                                              id_ + '.wav'

                            out_file.write('libri_' + id_ + '\t' + audio_file_path + '\t' + transcript+'\n')

    out_file.close()


if __name__ == '__main__':
    # needs two command line argument.
    # 1. root path of LibriSpeech
    # 2. output csv path
    main(sys.argv[1], sys.argv[2])
