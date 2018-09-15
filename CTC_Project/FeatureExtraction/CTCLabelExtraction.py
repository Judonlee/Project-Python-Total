import os

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\IEMOCAP-Transcription\\'
    savepath = 'D:\\ProjectData\\IEMOCAP-Label-Words\\'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                for indexD in ['ang', 'exc', 'sad', 'neu', 'hap']:
                    os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        print(indexA, indexB, indexC, indexD, indexE)
                        file = open(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                    'r')
                        data = file.read()
                        data = data.lower()
                        for indexX in range(len(data)):
                            if data[indexX] != ' ' and (ord(data[indexX]) < ord('a') or ord(data[indexX]) > ord('z')):
                                data = data.replace(data[indexX], ' ')

                        counter = 0
                        for sample in data.split(' '):
                            if sample != '':
                                counter += 1

                        file.close()

                        file = open(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                    'w')
                        if indexD == 'ang': appoint = 1
                        if indexD == 'exc' or indexD == 'hap': appoint = 2
                        if indexD == 'neu': appoint = 3
                        if indexD == 'sad': appoint = 4
                        for indexX in range(counter + 1):
                            if indexX != 0: file.write(',')
                            file.write(str(appoint))

                        file.close()
