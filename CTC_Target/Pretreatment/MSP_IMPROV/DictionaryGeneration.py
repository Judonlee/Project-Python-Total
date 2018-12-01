if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/'

    dictionary = {}
    with open(r'D:\ProjectData\MSP-IMPROVE\Evalution.txt', 'r') as file:
        data = file.readlines()
        for sample in data:
            if sample[0:3] == 'UTD':
                dictionary[sample[4:sample.find('.')]] = sample[sample.find(';') + 2]
        for sample in dictionary.keys():
            print(sample, dictionary[sample])

    with open(r'D:\ProjectData\MSP-IMPROVE\Dictionary.csv', 'w') as file:
        for sample in dictionary.keys():
            file.write(sample + ',' + dictionary[sample] + '\n')
