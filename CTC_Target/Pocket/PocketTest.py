import os
from pocketsphinx import AudioFile

audio = AudioFile(audio_file=r'C:\Users\BZT\Desktop\speech_segment\speech_segment\Ses01F_impro01_M013.wav',
                  keyphrase='yeah')
fps = 100
for phrase in audio:  # frate (default=100)
    # print('-' * 28)
    # print('| %5s |  %3s  |   %4s   |' % ('start', 'end', 'word'))
    # print('-' * 28)
    for s in phrase.seg():
        print('%4ss\t%4ss\t%8s' % (s.start_frame / fps, s.end_frame / fps, s.word))
    # print('-' * 28)

# from pocketsphinx import Pocketsphinx
#
# ps = Pocketsphinx(verbose=True, logfn='pocketsphinx.log')
# ps.decode()
#
# print(ps.hypothesis())
