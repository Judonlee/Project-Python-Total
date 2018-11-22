import os
import numpy

if __name__ == '__main__':
    with open(r'E:\LaboratoryData-Origin\FAU-AEC\transliteration\transliteration.txt', 'r') as file:
        data = file.readlines()

    savepath = 'D:/ProjectData/FAU-AEC-Treated/Transcription/'
    if not os.path.exists(savepath): os.makedirs(savepath)
    for sample in data:
        print(sample)
        with open(savepath + sample[0:sample.find(' ')] + '.txt', 'w') as file:
            file.write(sample[sample.find(' ') + 1:-3])
