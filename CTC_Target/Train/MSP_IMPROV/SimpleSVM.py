from sklearn.svm import SVC
import numpy
import os
import multiprocessing as mp
import time
from CTC_Target.Loader.MSP_Loader import Loader


def Treatment():
    loadpath = '/mnt/external/Bobs/CTC_Target_MSP/Feature/IS13-Npy/'
    for cSearch in range(12):
        for gammaSearch in range(-15, -3, 1):
            c = numpy.power(2, cSearch)
            gamma = 1 / numpy.power(2, -gammaSearch)
            print(c, gamma)
            savepath = 'IS13-Result/C=%f,Gamma=%f/' % (c, gamma)
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
    for index in range(6):
        p = mp.Process(target=Treatment)
        processList.append(p)
        p.start()
        time.sleep(5)
    for p in processList:
        p.join()
