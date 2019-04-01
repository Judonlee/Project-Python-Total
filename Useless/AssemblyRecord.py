import numpy

if __name__ == '__main__':
    dictionary = {}
    data = numpy.genfromtxt(fname='InputRecord-Assembly.csv', dtype=str, delimiter=',')

    counter = 0
    nameset = set()
    for index in range(len(data)):
        if int(data[index][3]) != counter:
            print(counter, nameset)
            counter = int(data[index][3])
            for sample in nameset:
                # print(sample)
                if sample in dictionary.keys():
                    dictionary[sample] += 1
                else:
                    dictionary[sample] = 1
            nameset.clear()
        nameset.add(data[index][0])

    print(nameset)

    for sample in nameset:
        # print(sample)
        if sample in dictionary.keys():
            dictionary[sample] += 1
        else:
            dictionary[sample] = 1

    for sample in dictionary.keys():
        print(sample, '\t', dictionary[sample])
