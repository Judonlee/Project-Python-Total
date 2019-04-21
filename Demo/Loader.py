import numpy as np


def Loader(appointSession, appointGender, datapath, labelpath):
    trainData, trainLabel, valData, valLabel, testData, testLabel = [], [], [], [], [], []

    for gender in ['F', 'M']:
        for session in range(1, 6):
            currentData = np.load(datapath + '%s-%d.npy' % (gender, session))
            currrentLabel = np.load(labelpath + '%s-%d.npy' % (gender, session))
            print(gender, session, np.shape(currentData), np.shape(currrentLabel))

            if session != appointSession:
                trainData.extend(currentData)
                trainLabel.extend(currrentLabel)
                continue

            if gender == appointGender:
                testData.extend(currentData)
                testLabel.extend(currrentLabel)
            else:
                valData.extend(currentData)
                valLabel.extend(currrentLabel)
    print(np.shape(trainData), np.shape(trainLabel), np.shape(valData), np.shape(valLabel), np.shape(testData),
          np.shape(testLabel))
    return trainData, trainLabel, valData, valLabel, testData, testLabel
