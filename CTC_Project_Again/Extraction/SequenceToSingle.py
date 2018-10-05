import numpy
import stats
import os


def Seq2Single(data):
    singleVector = []
    for indexY in range(numpy.shape(data)[1]):
        singleVector.append(numpy.mean(data[:, indexY]))
        singleVector.append(numpy.median(data[:, indexY]))
        singleVector.append(stats.quantile(data[:, indexY], p=0.25))
        singleVector.append(stats.quantile(data[:, indexY], p=0.75))
        singleVector.append(numpy.max(data[:, indexY]))
        singleVector.append(numpy.min(data[:, indexY]))
        singleVector.append(numpy.max(data[:, indexY]) - numpy.min(data[:, indexY]))
        singleVector.append(stats.quantile(data[:, indexY], p=0.75) - stats.quantile(data[:, indexY], p=0.25))
        singleVector.append(numpy.std(data[:, indexY]))
        singleVector.append(numpy.var(data[:, indexY]))
        singleVector.append(stats.skewness(data[:, indexY]))
        singleVector.append(stats.kurtosis(data[:, indexY]))
    return singleVector


if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\Project-CTC-Data\\Csv\\Bands120\\'
    savepath = 'D:\\ProjectData\\Project-CTC-Data\\Csv-Single\\Bands120\\'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        print(indexA, indexB, indexC, indexD, indexE)
                        data = numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')
                        file = open(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                    'w')
                        vector = Seq2Single(data=data)
                        for indexX in range(len(vector)):
                            if indexX != 0:
                                file.write(',')
                            file.write(str(vector[indexX]))
                        file.close()
                        # print(Seq2Single(data=data))
                        # print(numpy.shape(Seq2Single(data=data)))
                        # print(numpy.shape(data))
