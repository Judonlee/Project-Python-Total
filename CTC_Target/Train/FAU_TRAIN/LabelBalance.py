import numpy


def LabelBalance(trainData, trainLabel, trainSeq, trainScription):
    totalData, totalLabel, totalSeq, totalScription = [], [], [], []
    for labelAppoint in range(numpy.shape(trainLabel)[1]):
        data, label, seq, transcription = [], [], [], []
        for index in range(len(trainData)):
            if numpy.argmax(numpy.array(trainLabel[index])) == labelAppoint:
                data.append(trainData[index])
                label.append(trainLabel[index])
                seq.append(trainSeq[index])
                transcription.append(trainScription[index])
        # print(numpy.shape(data), numpy.shape(label), numpy.shape(seq), numpy.shape(transcription),
        #       numpy.sum(label, axis=0))

        if labelAppoint == 0: times = 6
        if labelAppoint == 1: times = 3
        if labelAppoint == 2: times = 1
        if labelAppoint == 3 or labelAppoint == 4: times = 8

        for _ in range(times):
            totalData.extend(data)
            totalLabel.extend(label)
            totalScription.extend(transcription)
            totalSeq.extend(seq)
    print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.shape(totalScription), numpy.shape(totalSeq),
          numpy.sum(totalLabel, axis=0))
    return totalData, totalLabel, totalSeq, totalScription
