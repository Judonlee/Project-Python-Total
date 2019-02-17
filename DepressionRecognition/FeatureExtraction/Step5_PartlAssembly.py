import os
import numpy

if __name__ == '__main__':
    datapath = 'D:/ProjectData/AVEC2017-Bands40/Step2_Csv_Normalization/'
    labelpath = 'D:/ProjectData/AVEC2017-Bands40/Step4_CMU_Label_Digital/'
    savepath = 'D:/ProjectData/AVEC2017-Bands40/Step5_Assembly/'

    for indexA in os.listdir(labelpath):
        os.makedirs(os.path.join(savepath, indexA))
        for indexB in os.listdir(os.path.join(labelpath, indexA)):
            totalData, totalLabel = [], []
            for indexC in os.listdir(os.path.join(labelpath, indexA, indexB)):
                if indexC.find('Participant') == -1: continue
                if not os.path.exists(os.path.join(datapath, indexA, indexB, indexC[0:indexC.find('.')] + '.csv')):
                    continue
                print(indexA, indexB, indexC)

                currentData = numpy.genfromtxt(
                    fname=os.path.join(datapath, indexA, indexB, indexC[0:indexC.find('.')] + '.csv'), dtype=float,
                    delimiter=',')
                currentLabel = numpy.genfromtxt(fname=os.path.join(labelpath, indexA, indexB, indexC), dtype=int,
                                                delimiter=',')

                if len(numpy.shape(currentData)) == 0 or len(numpy.shape(currentLabel)) == 0:
                    # os.system('PAUSE')
                    continue

                totalData.append(currentData)
                totalLabel.append(currentLabel)
            numpy.save(file=os.path.join(savepath, indexA, indexB + '_Data.npy'), arr=totalData)
            numpy.save(file=os.path.join(savepath, indexA, indexB + '_Label.npy'), arr=totalLabel)
