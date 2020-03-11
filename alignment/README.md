# Phoneme Alignment Using Gentle

Forced phoneme alignment was done using [Gentle](https://github.com/lowerquality/gentle), which is a robust yet lenient
aligner built on [Kaldi](https://github.com/kaldi-asr/kaldi). Gentle provides a python API as well as a REST API to 
force align audio files. Since, gentle contains a ton of platform dependent dependencies, it's convenient to use their
[pre-built Docker image](https://hub.docker.com/r/lowerquality/gentle/).

For this project, we built a docker image using the gentle docker image as base which can align audio files in batch
by reading input from a csv file.

To build the docker image run the following from `alignment` directory:
```shell script
$ docker build -t gentle-alignment .
``` 

By default we use [CMU](cmudict-0.7b.txt) dictionary,which is packed into the docker image. To add a new dict, 
modify the [`gentle_alignment.py`](gentle_alignment.py#L43) file, update [`Dockerfile`](Dockerfile) to package 
the new dict and rebuild the image.

Running the program requires four input parameters:
1. `input_csv`: A csv files containing the path of the wav file and it's transcript
2. `phoneme_path`: Output path where phoneme slices (wav files) would be written
3. `output_csv`: Path of the csv file which would contain metadata of the aligned file. This is needed when generating
training samples from the phoneme level wav files.
4. `wav_root`: The root path of the input wav files. If the `input_csv` contains the complete path of the input wav
file, this parameter can be set to a blank string

The `input_csv` is a tab separated file with three columns - id, wav_path & transcript and no header. A sample file
looks like this:
```tsv
libri_2582-155973-0032  train-clean-360/2582/155973/2582-155973-0032.wav        the little boy's feelings overcame him he had been loaned a king snake which as all nature lovers know is not only a useful but a beautiful snake very friendly to human beings and he came rushing home to show the treasure
libri_5724-13364-0083   train-clean-360/5724/13364/5724-13364-0083.wav  why need we care for outside things why indeed he said in a low fond tone so i easily found out how they meant to settle the difficulty namely
libri_2494-156015-0019  train-clean-360/2494/156015/2494-156015-0019.wav        they therefore remain bound the man who does not shrink from self crucifixion can never fail to accomplish the object upon which his heart is set this is as true of earthly as of heavenly things
libri_8193-116805-0037  train-clean-360/8193/116805/8193-116805-0037.wav        what shall we do wise medeia we must have water or we die of thirst flesh and blood we can face fairly but who can face this red hot brass i can face red hot brass if the tale i hear be true
libri_8066-290901-0040  train-clean-360/8066/290901/8066-290901-0040.wav        i promised him to come up soon but i continued on for some hours with the drunken crowd when i did come up to our apartment i found donald on his knees by his bed with his testament and an old hymn book of my mother in law's
```

The `output_csv` is also tab separated and has four columns - phoneme slice wav file name, id, word & phoneme
```tsv
libri_5808-54425-0000_five_ay1_690_820.wav      libri_5808-54425-0000   five    ay1
libri_5808-54425-0000_years_ih1_960_1070.wav    libri_5808-54425-0000   years   ih1
libri_5808-54425-0000_is_ih0_1300_1380.wav      libri_5808-54425-0000   is      ih0
libri_5808-54425-0000_but_ah1_1540_1590.wav     libri_5808-54425-0000   but     ah1
libri_5808-54425-0000_a_ey1_1650_1740.wav       libri_5808-54425-0000   a       ey1
libri_5808-54425-0000_short_ao1_1860_1950.wav   libri_5808-54425-0000   short   ao1
libri_5808-54425-0000_time_ay1_2230_2540.wav    libri_5808-54425-0000   time    ay1
libri_5808-54425-0000_in_ih1_2600_2670.wav      libri_5808-54425-0000   in      ih1
libri_5808-54425-0000_the_ah1_2750_2800.wav     libri_5808-54425-0000   the     ah1
libri_5808-54425-0000_life_ay1_2900_3000.wav    libri_5808-54425-0000   life    ay1
```

These inputs are given to the docker container via environment variables. An example of running alignment:
```shell script
$ docker run --rm -v /home/ubuntu/capstone:/work/capstone \
  -e input_csv=/work/capstone/gentle/libri_path_transcript_2.csv \
  -e phoneme_path=/work/capstone/gentle/phoneme_slices/ \
  -e output_csv=/work/capstone/gentle/aligned_phonemes_2.csv \
  -e wav_root=/work/capstone/gentle/LibriSpeech gentle-alignment:latest
```
Notice that a host directory is mounted on the container and all the paths are of the docker container and not the
host machine.

*Pro Tip: Alignment is not run in parallel inside of the docker container. To parallelize the process, split the
`input_csv` into multiple smaller files and then do a `docker run` for each of those files. Merge the `output_csv` 
files in the end.*
