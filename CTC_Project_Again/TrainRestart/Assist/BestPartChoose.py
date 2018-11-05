import numpy
import shutil
import os

if __name__ == '__main__':
    matrixPath = 'D:/ProjectData/BrandNewCTC/Data-Result-Changed-%s/Bands-%d-Session-%d-%s/Decode/%04d.csv'
    netpath = 'D:/ProjectData/BrandNewCTC/Data-Changed-%s/Bands-%d-Session-%d/'
    savepath = 'D:/ProjectData/BrandNewCTC/SingleBLSTM-Choosed/Bands-%d-Session-%d-%s/'
    bands = 30

    for session in range(1, 6):
        for gender in ['Female', 'Male']:
            maxUA, maxWA, uaMatrix, waMatrix = 0, 0, [], []
            uaPosition, waPosition = [], []
            for episode in range(100):
                for part in ['Left']:
                    matrix = numpy.genfromtxt(fname=matrixPath % (part, bands, session, gender, episode), dtype=float,
                                              delimiter=',')
                    WA, UA = 0, 0
                    for index in range(len(matrix)):
                        WA += matrix[index][index]
                        UA += matrix[index][index] / sum(matrix[index])
                    WA = WA / sum(sum(matrix))
                    UA = UA / len(matrix)
                    if UA >= maxUA:
                        maxUA = UA
                        uaMatrix = matrix.copy()
                        uaPosition = [part, episode]
                    if WA >= maxWA:
                        maxWA = WA
                        waMatrix = matrix.copy()
                        waPosition = [part, episode]
            print('Session %d Gender %s' % (session, gender))
            print(maxUA, maxWA, uaPosition, waPosition)
            print(uaMatrix)
            print(waMatrix)
            print('\n\n')

            os.makedirs(savepath % (bands, session, gender))
            shutil.copy(
                src=netpath % (uaPosition[0], bands, session) + '%04d-Network.data-00000-of-00001' % uaPosition[1],
                dst=savepath % (bands, session, gender) + 'UA.data-00000-of-00001')
            shutil.copy(
                src=netpath % (uaPosition[0], bands, session) + '%04d-Network.index' % uaPosition[1],
                dst=savepath % (bands, session, gender) + 'UA.index')
            shutil.copy(
                src=netpath % (uaPosition[0], bands, session) + '%04d-Network.meta' % uaPosition[1],
                dst=savepath % (bands, session, gender) + 'UA.meta')

            shutil.copy(
                src=netpath % (waPosition[0], bands, session) + '%04d-Network.data-00000-of-00001' % waPosition[1],
                dst=savepath % (bands, session, gender) + 'WA.data-00000-of-00001')
            shutil.copy(
                src=netpath % (waPosition[0], bands, session) + '%04d-Network.index' % waPosition[1],
                dst=savepath % (bands, session, gender) + 'WA.index')
            shutil.copy(
                src=netpath % (waPosition[0], bands, session) + '%04d-Network.meta' % waPosition[1],
                dst=savepath % (bands, session, gender) + 'WA.meta')
