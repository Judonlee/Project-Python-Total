import numpy
import os

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/FinalResult_ThreePart/MA_10_frame_100_Result/'
    savepath = 'E:/ProjectData_Depression/FinalResult_ThreePart/MA_10_frame_100_Result_Restart/'
    os.makedirs(savepath)
    for index in range(100):
        with open(loadpath + '%04d.csv' % index, 'r') as file:
            data = file.read()
        data = data.replace('[', '')
        data = data.replace(']', '')

        with open(savepath + '%04d.csv' % index, 'w') as file:
            file.write(data)
