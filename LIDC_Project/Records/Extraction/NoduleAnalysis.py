import os

if __name__ == '__main__':
    loadpath = 'E:\\LIDC\\LIDC-Nodules\\'

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

            if type in dictionary.keys():
                dictionary[type] += 1
            else:
                dictionary[type] = 1
    for key in dictionary.keys():
        print(key, dictionary[key])
