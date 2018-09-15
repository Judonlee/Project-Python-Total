import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
import os
import numpy


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def CTC_Loader_Part(wavname, filename):
    # Constants
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

    # Loading the data
    fs, audio = wav.read(wavname)
    inputs = mfcc(audio, samplerate=fs)

    # Tranform in 3D array
    train_inputs = np.asarray(inputs[np.newaxis, :])
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    train_seq_len = [train_inputs.shape[1]]

    # Readings targets
    with open(filename, 'r') as f:
        # Only the last line is necessary
        line = f.readlines()[-1]
        line = line.lower()
        for index in range(len(line)):
            if (ord(line[index]) > ord('z') or ord(line[index]) < ord('a')) and line[index] != ' ':
                line = line.replace(line[index], ' ')

        # Get only the words between [a-z] and replace period for none
        original = ' '.join(line.strip().lower().split(' ')).replace('.', '')
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')

    finaltarget = targets.copy()
    targets = [finaltarget[0]]
    for index in range(1, len(finaltarget)):
        if finaltarget[index - 1] == '' and finaltarget[index] == '':
            continue
        else:
            targets.append(finaltarget[index])
    print(targets)

    # Adding blank label
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from([targets])

    return train_inputs, train_targets, train_seq_len


def CTC_Loader(wavfold, transfold):
    totalTrainInputs, totalTrainTargets, totalSeqLen = [], [], []
    for indexA in os.listdir(wavfold):
        for indexB in os.listdir(wavfold + indexA):
            for indexC in os.listdir(wavfold + indexA + '\\' + indexB):
                for indexD in os.listdir(wavfold + indexA + '\\' + indexB + '\\' + indexC):
                    for indexE in os.listdir(wavfold + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        print(indexA, indexB, indexC, indexD, indexE)
                        # exit()
                        curTrainInputs, curTrainTargets, curSeqLen = CTC_Loader_Spectrogram_Part(
                            dataname=wavfold + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            filename=transfold + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\'
                                     + indexE[0:indexE.find('.')] + '.txt')
                        if curSeqLen == 0:
                            continue
                        totalTrainInputs.append(curTrainInputs)
                        totalTrainTargets.append(curTrainTargets)
                        totalSeqLen.append(curSeqLen)
    return totalTrainInputs, totalTrainTargets, totalSeqLen


def CTC_Loader_Spectrogram_Part(dataname, filename):
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    FIRST_INDEX = ord('a') - 1

    inputs = numpy.genfromtxt(fname=dataname, dtype=float, delimiter=',')
    # inputs = numpy.transpose(a=inputs, axes=(1, 0))
    train_inputs = np.asarray(inputs[np.newaxis, :])
    train_seq_len = [train_inputs.shape[1]]

    # Readings targets
    with open(filename, 'r') as f:
        # Only the last line is necessary
        line = f.readlines()[-1]
        line = line.lower()
        for index in range(len(line)):
            if (ord(line[index]) > ord('z') or ord(line[index]) < ord('a')) and line[index] != ' ':
                line = line.replace(line[index], ' ')

        # Get only the words between [a-z] and replace period for none
        original = ' '.join(line.strip().lower().split(' ')).replace('.', '')
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')

    finaltarget = targets.copy()
    targets = [finaltarget[0]]
    for index in range(1, len(finaltarget)):
        if finaltarget[index - 1] == '' and finaltarget[index] == '':
            continue
        else:
            targets.append(finaltarget[index])
    print(targets)

    # Adding blank label
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
    print(targets)

    # Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])
    print(targets)

    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from([targets])
    print(train_targets)
    print(len(targets))
    if len(targets) > 20: return 0, 0, 0
    return train_inputs, train_targets, train_seq_len


if __name__ == '__main__':
    data = numpy.genfromtxt(
        r'D:\ProjectData\IEMOCAP-Treated\Bands-60-csv\improve\Female\Session1\neu\Ses01F_impro01_F000.wav.csv',
        dtype=float, delimiter=',')
    print(numpy.shape(data))
