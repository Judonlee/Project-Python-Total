import numpy
import os


def Loader(loadpath, testSession):
    '''
    :param loadpath:读取文件的路径
    :param testSession: 选择哪一个session为测试集（该部分不被纳入训练集之中）
    :return: 训练集的数据、标签、长度；测试集的数据、标签、长度
    '''
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            # 文件已经压缩好了，直接读取就行了
            data = numpy.load(os.path.join(loadpath, '%s-Session%d-Data.npy' % (gender, session)))
            label = numpy.load(os.path.join(loadpath, '%s-Session%d-Label.npy' % (gender, session)))
            seq = numpy.load(os.path.join(loadpath, '%s-Session%d-Seq.npy' % (gender, session)))
            print('Loading Session%s - %s\t' % (session, gender), numpy.shape(data), numpy.shape(label),
                  numpy.shape(seq))

            if session != testSession:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
            else:
                testData.extend(data)
                testLabel.extend(label)
                testSeq.extend(seq)
    print('\nTrainData:', numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq),
          numpy.sum(trainLabel, axis=0))
    print('TestData:', numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq),
          numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


if __name__ == '__main__':
    loadpath = 'E:/CTC_Target/Features/Bands30/'
    Loader(loadpath=loadpath, testSession=0)
