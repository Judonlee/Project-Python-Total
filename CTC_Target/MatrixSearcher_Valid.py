import numpy
import os

if __name__ == '__main__':

    data = numpy.array([[11.7, 2.7, 1.2, 1.4, 3.3],
                        [2.7, 12.5, 2.7, 0.8, 1.6],
                        [1.7, 3.0, 9.9, 1.5, 2.5],
                        [1.3, 1.3, 2.5, 10.3, 4.4],
                        [2.7, 2.0, 1.9, 4.8, 9.7]])
    WA, UA = 0, 0
    for index in range(len(data)):
        WA += data[index][index]
        UA += data[index][index] / sum(data[index])

    # print(sum(data))
    WA = WA / sum(sum(data))
    UA = UA / len(data)
    print(WA, UA)
