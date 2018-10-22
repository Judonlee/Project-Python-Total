import os
import shutil

if __name__ == '__main__':
    bands = 120
    UAepisode = [84, 35, 43, 6, 28, 29, 1, 33, 14, 67, ]
    WAepisode = [56, 34, 87, 6, 79, 70, 28, 72, 23, 59, ]
    for appoint in range(9, 10):
        loadpath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-CMU-New/Bands-%d-%d/' % (bands, appoint)
        savepath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-CMU-New-Choosed/Bands-%d-%d/' % (bands, appoint)

        if os.path.exists(savepath):
            print('Have Already Done')
            exit()
        if not os.path.exists(savepath): os.makedirs(savepath)

        for filename in os.listdir(loadpath):
            if filename[0:4] == '%04d' % WAepisode[appoint]:
                shutil.copy(loadpath + filename, savepath + 'WA' + filename[filename.find('.'):])
                print(filename)
            if filename[0:4] == '%04d' % UAepisode[appoint]:
                shutil.copy(loadpath + filename, savepath + 'UA' + filename[filename.find('.'):])
                print(filename)
