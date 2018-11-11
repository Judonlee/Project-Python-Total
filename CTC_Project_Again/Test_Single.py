import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP-New-Again/Bands30/%s-Session%d-%s.npy'
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            data = numpy.load(loadpath % (gender, session, 'Data'))
            scription = numpy.load(loadpath % (gender, session, 'Scription'))
            seq = numpy.load(loadpath % (gender, session, 'Seq'))
            for index in range(len(data)):
                # print(numpy.shape(data[index]), len(scription[index]))
                # if numpy.shape(data[index])[0] < len(scription[index]):
                #     print('Find It!!!')
                # if seq[index] < len(scription[index]):
                #     print('Find IT')

                # print(numpy.shape(data[index]), seq[index])
                if numpy.shape(data[index])[0]!=seq[index]:
                    print('Find IT')