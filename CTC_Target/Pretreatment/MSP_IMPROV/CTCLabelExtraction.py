import os

if __name__ == '__main__':
    dictionary = {}
    with open('D:/GitHub/CTC_Target/Pretreatment/MSP_IMPROV/Dictionary.txt', 'r') as file:
        data = file.readlines()
        for sample in data:
            dictionary[sample.split('  ')[0]] = sample[sample.find('  ') + 2:-1]
    # for sample in dictionary.keys():
    #     print(sample, dictionary[sample])

    loadpath = 'D:/ProjectData/MSP-IMPROVE/Transcription/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/Transcription-CMU/'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                os.makedirs(os.path.join(savepath, indexA, indexB, indexC))
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    with open(os.path.join(loadpath, indexA, indexB, indexC, indexD), 'r') as file:
                        data = file.read()
                        data = data.upper()
                        data = data.replace(',', '')
                        data = data.replace('?', '')
                        data = data.replace('.', '')
                        data = data.replace('!', '')
                        data = data.replace('\n', '')
                        data = data.replace('\r', '')
                        data = data.replace('\t', '')
                        # print(data)
                        # print(dictionary['ABOUT'])
                        with open(os.path.join(savepath, indexA, indexB, indexC, indexD), 'w') as file:
                            for sample in data.split(' '):
                                if sample in dictionary.keys():
                                    file.write(dictionary[sample] + ' ')
                            # print(dictionary[sample])

                    # exit()
                    print(indexA, indexB, indexC, indexD)
