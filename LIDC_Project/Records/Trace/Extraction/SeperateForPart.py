import os
import numpy
import shutil

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/FinalResult/Nodules/'
    savepath = 'E:/LIDC/TreatmentTrace/FinalResult/Part4/Nodules/'
    counter = 0
    # for indexA in os.listdir(loadpath):
    #     for indexB in os.listdir(os.path.join(loadpath, indexA)):
    #         if counter % 5 == 4:
    #             print(indexA, indexB)
    #             if not os.path.exists(os.path.join(savepath, indexA)): os.makedirs(os.path.join(savepath, indexA))
    #             shutil.copy(os.path.join(loadpath, indexA, indexB), os.path.join(savepath, indexA, indexB))
    #         counter += 1

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                if counter % 5 == 4:
                    print(indexA, indexB, indexC)
                    if not os.path.exists(os.path.join(savepath, indexA, indexB)): os.makedirs(
                        os.path.join(savepath, indexA, indexB))
                    shutil.copy(os.path.join(loadpath, indexA, indexB, indexC),
                                os.path.join(savepath, indexA, indexB, indexC))
                counter += 1
