import os

if __name__ == '__main__':
    loadpath = 'F:\\LIDC\\LIDC-Nodules\\'

    dictionary = {}
    search = '<malignancy>'
    searchEnd = '</malignancy>'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            print(indexA, indexB)
            file = open(loadpath + indexA + '\\' + indexB + '\\Character.txt', 'r')
            data = file.read()
            file.close()
            type = data[data.find(search):data.find(searchEnd)]

            if type in dictionary.keys():
                dictionary[type] += 1
            else:
                dictionary[type] = 1
    for key in dictionary.keys():
        print(key, dictionary[key])
