import os
import shutil

if __name__ == '__main__':
    loadpath = 'E:\\LIDC\\LIDC-Nodules\\'
    savepath = 'E:\\LIDC\\LIDC-Nodules-Selected\\'

    dictionary = {}
    search = '<texture>'
    searchEnd = '</texture>'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            print(indexA, indexB)
            file = open(loadpath + indexA + '\\' + indexB + '\\Character.txt', 'r')
            data = file.read()
            file.close()

            preLimited = '<internalStructure>'
            preLimitedEnd = '</internalStructure>'
            preType = data[data.find(preLimited) + len(preLimited):data.find(preLimitedEnd)]
            if preType != '1': continue

            preLimited = '<calcification>'
            preLimitedEnd = '</calcification>'
            preType = data[data.find(preLimited) + len(preLimited):data.find(preLimitedEnd)]
            if preType != '6': continue

            type = data[data.find(search):data.find(searchEnd)]
            if type != search + '5': continue

            if not os.path.exists(savepath + indexA):
                os.makedirs(savepath + indexA)
            shutil.copytree(loadpath + indexA + '\\' + indexB, savepath + indexA + '\\' + indexB)
