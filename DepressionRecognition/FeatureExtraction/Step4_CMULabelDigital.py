import os
import numpy

dictionary = {'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10, 'EH': 11,
              'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22,
              'N': 23, 'NG': 24, 'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30, 'T': 31, 'TH': 32, 'UH': 33,
              'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39}

print(dictionary['AA'])
if __name__ == '__main__':
    loadpath = 'D:/ProjectData/AVEC2017-Bands40/Step3_CMU_Label/'
    savepath = 'D:/ProjectData/AVEC2017-Bands40/Step4_CMU_Label_Digital/'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            os.makedirs(os.path.join(savepath, indexA, indexB))
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                print(indexA, indexB, indexC)

                with open(os.path.join(loadpath, indexA, indexB, indexC), 'r') as file:
                    data = file.read()
                with open(os.path.join(savepath, indexA, indexB, indexC + '.csv'), 'w') as file:
                    for index in range(len(data.split(' ')) - 1):
                        sample = data.split(' ')[index]
                        if sample not in dictionary.keys():
                            continue
                        print(dictionary[sample], end=' ')
                        if index != 0: file.write(',')
                        file.write(str(dictionary[sample]))
                    print()
                # exit()
