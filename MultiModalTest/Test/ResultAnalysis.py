import os
import numpy

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-Transform-LA-3-Punishment-100000-Result/Bands30/'
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            if not os.path.exists(os.path.join(loadpath, 'Session%d-%s' % (session, gender))):
                print()
                continue
            uaList, waList = [], []
            for episode in range(199):
                data = numpy.genfromtxt(
                    fname=os.path.join(loadpath, 'Session%d-%s' % (session, gender), '%04d-Decode.csv' % episode),
                    dtype=int, delimiter=',')

                uaCounter, waCounter = 0, 0
                for index in range(len(data)):
                    uaCounter += data[index][index] / sum(data[index])
                    waCounter += data[index][index]
                uaCounter /= len(data)
                waCounter /= sum(sum(data))
                # print(uaCounter, waCounter)
                uaList.append(uaCounter)
                waList.append(waCounter)
            print(max(uaList), '\t', max(waList))
