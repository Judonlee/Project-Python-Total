import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Seq-Features/IS13/improve/Female/Session1/ang/Ses01F_impro01_F012.wav.csv'
    totalData = numpy.genfromtxt(loadpath, dtype=float, delimiter=',')
    print(numpy.shape(totalData))
