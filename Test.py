import re

if __name__ == '__main__':
    with open('English.txt', 'r') as file:
        data = file.readlines()

    with open('EnglishOutputMedia.txt', 'w') as file:
        for sample in data:
            resample = ''
            flag = True
            for index in range(len(sample)):
                if sample[index] == '(': flag = False
                if flag: resample += sample[index]
                if sample[index] == ')': flag = True
            sample = resample
            for index in range(len(sample)):
                if index == 0:
                    file.write(sample[index])
                    continue
                if sample[index] == sample[index - 1] and sample[index] == '\t': continue
                if sample[index] == '\t':
                    file.write('\n')
                    continue
                file.write(sample[index])

    with open('EnglishOutputMedia.txt', 'r') as file:
        data = file.readlines()

    with open('EnglishOutput.txt', 'w') as file:
        for indexX in range(len(data)):
            startFlag = True
            for indexY in range(len(data[indexX])):
                if startFlag and data[indexX][indexY] == ' ': continue
                if startFlag and data[indexX][indexY] == '\n': continue
                if data[indexX][indexY] != ' ': startFlag = False
                if not startFlag: file.write(data[indexX][indexY])
