import os
import numpy

if __name__ == '__main__':

    dictionary = {}
    with open('D:/GitHub/CTC_Target/Pretreatment/MSP_IMPROV/Dictionary.txt', 'r') as file:
        data = file.readlines()
        for sample in data:
            if sample.split('  ')[0][-1] == ')': continue
            sample = sample.replace('0', '')
            sample = sample.replace('1', '')
            sample = sample.replace('2', '')
            dictionary[sample.split('  ')[0]] = sample[sample.find('  ') + 2:-1]
    # for sample in dictionary.keys():
    #     print(sample, dictionary[sample])

    loadpath = 'D:/ProjectData/AVEC2017-Separate/'
    savepath = 'D:/ProjectData/AVEC2017-Bands40/Step3_CMU_Label/'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            os.makedirs(os.path.join(savepath, indexA, indexB))
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                if indexC[-3:] != 'csv': continue
                print(indexA, indexB, indexC)

                with open(os.path.join(loadpath, indexA, indexB, indexC), 'r') as file:
                    data = file.read()

                chooseResult = ''
                for sample in data:
                    sample = sample.upper()
                    if ('A' <= sample) and (sample <= 'Z'):
                        chooseResult += sample
                    if sample in [' ', '\'']:
                        chooseResult += sample
                # print(data)
                # print(chooseResult)

                with  open(os.path.join(savepath, indexA, indexB, indexC + '.txt'), 'w') as file:
                    for sample in chooseResult.split(' '):
                        if sample in dictionary.keys():
                            print(dictionary[sample], end=' ')
                            file.write(dictionary[sample] + ' ')
                    print()
                # exit()
