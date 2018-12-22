import matplotlib.pylab as plt
import numpy
import os

THRESHOLD = 500

if __name__ == '__main__':
    bands = 100
    loadfile = 'E:/OneUseData/FAU/bands-%d/mont-std.npy' % bands
    labelfile = 'E:/OneUseData/FAU/bands-%d/mont-label.npy' % bands
    savepath = 'D:/OneUseData/FAU-Png/Bands%d-224/Mont/' % bands

    counter = 0
    data = numpy.load(loadfile)
    label = numpy.load(labelfile)
    for index in range(len(data)):
        counter += 1
        sample = data[index].copy()
        print(numpy.shape(sample))

        if numpy.shape(sample)[0] < THRESHOLD:
            sample = numpy.concatenate(
                (sample, numpy.zeros((THRESHOLD - numpy.shape(sample)[0], numpy.shape(sample)[1]))), axis=0)
        else:
            sample = sample[0:500]
        # print(numpy.shape(sample))

        if not os.path.exists(os.path.join(savepath, str(numpy.argmax(label[index])))):
            os.makedirs(os.path.join(savepath, str(numpy.argmax(label[index]))))

        plt.figure(figsize=(numpy.shape(sample)[1] / 100, numpy.shape(sample)[0] / 100))
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.imshow(sample)
        plt.savefig(os.path.join(savepath, str(numpy.argmax(label[index])), '%d.current.png' % counter))
        plt.clf()
        plt.close()

        sample = plt.imread(os.path.join(savepath, str(numpy.argmax(label[index])), '%d.current.png' % counter))
        sample = plt.resize(sample, [224, 224])
        plt.figure(figsize=(numpy.shape(sample)[1] / 100, numpy.shape(sample)[0] / 100))
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.imshow(sample)
        plt.savefig(os.path.join(savepath, str(numpy.argmax(label[index])), '%d.png' % counter))
        plt.clf()
        plt.close()

        os.remove(os.path.join(savepath, str(numpy.argmax(label[index])), '%d.current.png' % counter))
        # break
