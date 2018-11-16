import os
import pydicom

if __name__ == '__main__':
    loadpath = 'E:/LIDC/LIDC-IDRI/'
    savepath = 'E:/LIDC/InstanceNumber/'
    # os.makedirs(savepath)
    for indexA in os.listdir(loadpath):
        file = open(savepath + indexA + '.csv', 'w')
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    if indexD[-3:] != 'dcm': continue
                    print(indexA, indexD)

                    DCMFile = pydicom.read_file(os.path.join(loadpath, indexA, indexB, indexC, indexD))
                    file.write(str(DCMFile.InstanceNumber) + ',' + str(DCMFile.SOPInstanceUID) + '\n')
        file.close()
