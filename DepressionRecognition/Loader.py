import numpy
import os


def Load_EncoderDecoder():
    loadpath = 'D:/ProjectData/AVEC2017-Bands40/Step5_Assembly/'
    for indexA in ['Train', 'Develop', 'Test']:
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            if indexB.find('Data') == -1: continue
            print(indexA, indexB)


if __name__ == '__main__':
    Load_EncoderDecoder()
