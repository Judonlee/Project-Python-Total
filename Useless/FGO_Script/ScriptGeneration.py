import numpy

if __name__ == '__main__':
    orders = numpy.genfromtxt('Input.txt', dtype=str, delimiter=',')
    origin = numpy.genfromtxt('Parameter.csv', dtype=str, delimiter=',')
    parameter = {}
    for sample in origin:
        if sample[0].find('Time') == -1:
            parameter[sample[0]] = sample[1] + ' ' + sample[2]
        else:
            parameter[sample[0]] = sample[1]

    # for sample in parameter.keys():
    #     print(sample, parameter[sample])
    with open('Output.txt', 'w') as file:
        file.write('#delay 1000\n')
        for sample in orders:
            if sample == '': continue
            if sample.find('Time') == -1 and sample.find('Clothes') == -1:
                file.write('#click %s\n' % parameter[sample])
            if sample == 'Initial Position' or sample == 'Scene Time':
                file.write('#delay %s\n' % parameter['Scene Time'])
            if sample == 'Start Battle' or sample.find('Skill') != -1:
                file.write('#delay %s\n' % parameter['Skill Time'])
            if sample.find('Noble') != -1 or sample.find('Action') != -1:
                file.write('#delay %s\n' % parameter['Action Time'])
            if sample.find('Clothes') != -1:
                file.write('#click %s\n' % parameter['Clothes Initial'])
                file.write('#delay %s\n' % parameter['Skill Time'])
                file.write('#click %s\n' % parameter[sample])
                file.write('#delay %s\n' % parameter['Skill Time'])
            print(sample)

    print('\n\n')
    with open('Output.txt', 'r') as file:
        for sample in file.readlines():
            print(sample, end='')
