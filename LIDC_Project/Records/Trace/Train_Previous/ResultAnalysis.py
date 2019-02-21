import numpy
from LIDC_Project.Trace.Train_Previous.Tools import AUC_Calculation, Result_Calculation
import xlwt
import matplotlib.pylab as plt
import os

MATRIX_START_X = 4
MATRIX_START_Y = 7

if __name__ == '__main__':
    searchPath = 'E:/LIDC/TreatmentTrace/Wavelet_db4/'
    for part in os.listdir(searchPath):
        loadpath = searchPath + part + '/'
        savepath = 'E:/LIDC/TreatmentTrace/Wavelet_db4_Result/'
        if not os.path.exists(savepath): os.makedirs(savepath)
        savename = part

        totalResult, matrixList = [], []
        for appoint in range(10):
            testLabel = numpy.load('E:/LIDC/TreatmentTrace/Step7-TotalNpy/OriginCsv/Part%d-Label.npy' % appoint)
            probability = numpy.genfromtxt(fname=loadpath + 'Batch%d.csv' % appoint, dtype=float, delimiter=',')
            testLabel = numpy.argmax(testLabel, axis=1)
            auc = AUC_Calculation(testLabel=testLabel, probability=probability, figureFlag=True, showFlag=False,
                                  legend='Batch-%d' % appoint)
            result, matrix = Result_Calculation(testLabel=testLabel, probability=probability)
            result.append(auc)
            totalResult.append(result)
            matrixList.append(matrix)
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.legend()
        plt.title(savename)
        plt.xlabel('fpr (%)')
        plt.ylabel('tpr (%)')
        plt.savefig(savepath + savename + '.png')
        plt.clf()
        plt.close()
        # plt.show()

        # print(totalResult)
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('test', cell_overwrite_ok=True)

        sheet.write_merge(0, 0, 0, 4, savename)
        sheet.write(1, 1, 'Precision')
        sheet.write(1, 2, 'Sensitivity')
        sheet.write(1, 3, 'Specificity')
        sheet.write(1, 4, 'AUC')
        for indexX in range(len(totalResult)):
            sheet.write(indexX + 2, 0, 'Batch-%d' % indexX)
            for indexY in range(len(totalResult[indexX])):
                sheet.write(indexX + 2, 1 + indexY, totalResult[indexX][indexY])
        sheet.write(len(totalResult) + 2, 0, 'Average')
        sheet.write(len(totalResult) + 2, 1, numpy.average(totalResult, axis=0)[0])
        sheet.write(len(totalResult) + 2, 2, numpy.average(totalResult, axis=0)[1])
        sheet.write(len(totalResult) + 2, 3, numpy.average(totalResult, axis=0)[2])
        sheet.write(len(totalResult) + 2, 4, numpy.average(totalResult, axis=0)[3])

        for index in range(len(matrixList)):
            xPosition = MATRIX_START_X + (numpy.shape(matrixList)[1] + 2) * int(index / 5)
            yPosition = MATRIX_START_Y + (numpy.shape(matrixList)[1] + 2) * int(index % 5)

            sheet.write_merge(xPosition - 1, xPosition - 1, yPosition,
                              yPosition + numpy.shape(matrixList)[1] - 1, 'Batch-%d' % index)
            for indexX in range(numpy.shape(matrixList)[1]):
                for indexY in range(numpy.shape(matrixList)[2]):
                    sheet.write(xPosition + indexX, yPosition + indexY, matrixList[index][indexX][indexY])

            book.save(savepath + savename + '.xls')
