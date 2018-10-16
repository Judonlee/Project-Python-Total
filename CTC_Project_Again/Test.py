import os
import shutil

if __name__ == '__main__':
    bands = 120
    WAepisode = [40, 35, 48, 11, 35, 32, 23, 18, 11, 23, ]
    UAepisode = [85, 35, 27, 10, 35, 9, 74, 58, 3, 12, ]
    for appoint in range(10):
        loadpath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-Class5-LR1E-3-RMSP/Bands-%d-%d/' % (bands, appoint)
        savepath = 'D:/ProjectData/Project-CTC-Data/NetworkParameter-CTC-Class5/Bands-%d-%d/' % (bands, appoint)

        if os.path.exists(savepath):
            print('Have Already Done')
            exit()
        if not os.path.exists(savepath): os.makedirs(savepath)

        for filename in os.listdir(loadpath):
            if filename[0:4] == '%04d' % WAepisode[appoint]:
                shutil.copy(loadpath + filename, savepath + 'WA' + filename[filename.find('.'):])
            if filename[0:4] == '%04d' % UAepisode[appoint]:
                shutil.copy(loadpath + filename, savepath + 'UA' + filename[filename.find('.'):])
