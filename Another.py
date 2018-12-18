import numpy

if __name__ == '__main__':
    for bands in [30, 40, 60, 80, 100]:
        data = numpy.genfromtxt('Bands%s.csv' % bands, dtype=float, delimiter=',')
        # print(data)
        WA, UA = 0, 0

        for index in range(numpy.shape(data)[0]):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
            # print(UA)
        WA /= sum(sum(data))
        UA /= numpy.shape(data)[0]
        print(WA, '\t', UA)
