import numpy
import os

THRESHOLD = 32

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step3-NoduleMedia/'
    savepath = 'E:/LIDC/TreatmentTrace/Step4-FinalDecision/'
    os.makedirs(savepath)

    for filename in os.listdir(loadpath):
        filedata = numpy.reshape(numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=str, delimiter=','),
                                 newshape=[-1, 4])
        if len(filedata) == 0: continue
        # print(filedata)
        decideNodules = [[filedata[0][0], float(filedata[0][1]), float(filedata[0][2]), float(filedata[0][3]), 1]]

        for index in range(1, len(filedata)):
            position = [float(filedata[index][1]), float(filedata[index][2]), float(filedata[index][3])]

            flag = True
            for comparison in decideNodules:
                # print(comparison)
                distance = numpy.sum(numpy.abs(numpy.subtract(position, comparison[1:-1])))
                if distance < THRESHOLD:
                    comparison[-1] += 1
                    flag = False
                    break
            if flag: decideNodules.append(
                [filedata[index][0], float(filedata[index][1]), float(filedata[index][2]), float(filedata[index][3]),
                 1])

        # print(decideNodules)
        with open(os.path.join(savepath, filename), 'w') as file:
            for sample in decideNodules:
                for index in range(len(sample)):
                    if index != 0: file.write(',')
                    file.write(str(sample[index]))
                file.write('\n')

        # exit()
