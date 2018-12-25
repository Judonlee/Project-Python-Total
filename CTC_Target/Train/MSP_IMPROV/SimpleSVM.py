from sklearn.svm import SVC
import numpy
import os
import multiprocessing as mp
import time


def Loader(loadpath, appointSession, appointGender):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for session in range(1, 7):
        for gender in ['F', 'M']:
            data = numpy.load(loadpath + 'Session%d-%s-Data.npy' % (session, gender))
            label = numpy.load(loadpath + 'Session%d-%s-Label.npy' % (session, gender))
            print('Session%d-%s' % (session, gender), numpy.shape(data), numpy.shape(label))

            if appointSession == session and appointGender == gender:
                testData.extend(data)
                testLabel.extend(label)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0), numpy.shape(testData),
          numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


def Treatment():
    loadpath = 'D:/ProjectData/MSP-IMPROVE/OpenSmile/IS12-Npy/'
    for cSearch in range(12):
        for gammaSearch in range(-15, -3, 1):
            c = numpy.power(2, cSearch)
            gamma = 1 / numpy.power(2, -gammaSearch)
            print(c, gamma)
            savepath = 'D:/ProjectData/MSP-IMPROVE/OpenSmile/IS12-Result/C=%f,Gamma=%f/' % (c, gamma)
            if not os.path.exists(savepath): os.makedirs(savepath)
            for appointSession in range(1, 7):
                for appointGender in ['F', 'M']:
                    if os.path.exists(savepath + 'Session-%d-%s.csv' % (appointSession, appointGender)):
                        continue
                    with open(savepath + 'Session-%d-%s.csv' % (appointSession, appointGender), 'w'):
                        pass
                    trainData, trainLabel, testData, testLabel = Loader(loadpath=loadpath,
                                                                        appointSession=appointSession,
                                                                        appointGender=appointGender)
                    trainLabel = numpy.argmax(trainLabel, axis=1)
                    testLabel = numpy.argmax(testLabel, axis=1)

                    clf = SVC(C=c, gamma=gamma)
                    clf.fit(trainData, trainLabel)
                    print('Train Completed')
                    result = clf.predict(testData)

                    matrix = numpy.zeros((4, 4))
                    for index in range(len(result)):
                        matrix[testLabel[index]][result[index]] += 1
                    print(matrix)

                    with open(savepath + 'Session-%d-%s.csv' % (appointSession, appointGender), 'w') as file:
                        for indexA in range(numpy.shape(matrix)[0]):
                            for indexB in range(numpy.shape(matrix)[1]):
                                if indexB != 0: file.write(',')
                                file.write(str(matrix[indexA][indexB]))
                            file.write('\n')


if __name__ == '__main__':
    processList = []
    for index in range(10):
        p = mp.Process(target=Treatment)
        processList.append(p)
        p.start()
        time.sleep(5)
    for p in processList:
        p.join()
