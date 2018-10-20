import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU/'
    counter = 0

    dict = {}
    for indexA in ['improve']:
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    for indexE in os.listdir(os.path.join(loadpath, indexA, indexB, indexC, indexD)):
                        # print(indexA, indexB, indexC, indexD, indexE)
                        if indexD == 'exc': indexD = 'hap'
                        key = '%s-%s-%s' % (indexB, indexC, indexD)
                        if key in dict:
                            dict[key] += 1
                        else:
                            dict[key] = 1
    # print(counter)
    for sample in dict.keys():
        print(sample, dict[sample])
