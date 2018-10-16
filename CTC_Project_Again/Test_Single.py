import os
import shutil

if __name__ == '__main__':
    bands = 40
    WAepisode = 53
    UAepisode = 25
    appoint = 5
    loadpath = 'D:/ProjectData/Project-CTC-Data/Records-CTC-Class5-LR1E-3-RMSP/Bands-%d-%d/' % (bands, appoint)
    savepath = 'D:/ProjectData/Project-CTC-Data/NetworkParameter-CTC-Class5/Bands-%d-%d/' % (bands, appoint)

    if os.path.exists(savepath):
        print('Have Already Done')
        exit()
    if not os.path.exists(savepath): os.makedirs(savepath)

    for filename in os.listdir(loadpath):
        if filename[0:4] == '%04d' % WAepisode:
            shutil.copy(loadpath + filename, savepath + 'WA' + filename[filename.find('.'):])
        if filename[0:4] == '%04d' % UAepisode:
            shutil.copy(loadpath + filename, savepath + 'UA' + filename[filename.find('.'):])
