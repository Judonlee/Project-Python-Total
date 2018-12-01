import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/Voice/'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            print(indexA, '\t', indexB, end='\t')
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                print(len(os.listdir(os.path.join(loadpath, indexA, indexB, indexC))), end='\t')
            print()
