import os

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\OneUse\\AVEC-SVR-Changed\\IS13\\'
    for index in os.listdir(loadpath):
        file = open(loadpath + index, 'r')
        data = file.read()
        file.close()
        print(index, ',', data)
