import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU/improve/'

    dictionary = {}
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    print(indexA, indexB, indexC, indexD)

                    with open(os.path.join(loadpath, indexA, indexB, indexC, indexD), 'r') as file:
                        data = file.read()
                        for sample in data.split(' '):
                            appoint = sample
                            if len(sample) < 1: continue
                            # print(sample[-1])
                            if sample[-1] >= '0' and sample[-1] <= '9':
                                appoint = sample[0:-1]

                            if appoint in dictionary.keys():
                                dictionary[appoint] += 1
                            else:
                                dictionary[appoint] = 1
    # print(counter)
    print(dictionary)

    for sample in dictionary.keys():
        print(sample, '\t', dictionary[sample])
