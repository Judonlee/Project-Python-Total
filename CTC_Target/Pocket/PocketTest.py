from pocketsphinx import AudioFile

fps = 100
audio = AudioFile(
    audio_file=r'D:\ProjectData\IEMOCAP\IEMOCAP-Voices\improve\Female\Session1\ang\Ses01F_impro01_F012.wav')
print('%4s\t%4s\t%4s' % ('Words', 'Start', 'End'))
for phrase in audio:
    print(phrase)
    # for s in phrase.seg():
    #     # print('| %4ss | %4ss | %8s |' % (s.start_frame / fps, s.end_frame / fps, s.word))
    #     # print(s.word, '\t', s.start_frame/100, 's\t', s.end_frame/100,'s')
    #     print('%4s\t%.2fs\t%.2fs' % (s.word, s.start_frame / 100, s.end_frame / 100))
