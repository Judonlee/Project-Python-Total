from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_SeqLabelLoader
import numpy
import os

if __name__ == '__main__':
    data = numpy.genfromtxt(
        r'D:\ProjectData\IEMOCAP\IEMOCAP-Features\GeMAPS\improve\Female\Session1\ang\Ses01F_impro01_F012.wav.csv',
        dtype=float, delimiter=',')
    print(numpy.shape(data))
