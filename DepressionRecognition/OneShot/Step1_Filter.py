import os
import numpy
import shutil

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/AVEC2017/depression/'
    savepath = 'D:/ProjectData/AVEC2017/Step1_Filter/'

    for part in ['train', 'dev', 'test']:
        os.makedirs(savepath + part)

        data = numpy.genfromtxt(fname=loadpath + '%s_split_Depression_AVEC2017.csv' % part, dtype=str, delimiter=',')
        for index in range(1, numpy.shape(data)[0]):
            print('Copying %s %s' % (part, data[index][0]))
            shutil.copytree(src=os.path.join(loadpath, '%s_P' % data[index][0]),
                            dst=os.path.join(savepath, part, '%s_P' % data[index][0]),
                            ignore=shutil.ignore_patterns('*.txt', '*.bin', '*_COVAREP.csv', '*_FORMANT.csv'))
