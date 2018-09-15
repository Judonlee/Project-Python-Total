import numpy as np
import pandas as pd
from enum import Enum
from keras.utils import to_categorical
from utils.model_utils import FeatureSets

base_path = '/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/shared/FAU-Aibo'


class FAUDataSets(Enum):
    school = 'school'
    binary = 'binary'


def load_fau_3_fold():
    """
    Load FAU-Aibo data set with classical three fold validation.

    :return: train, test, validation data sets.
    """
    folds = [
        {
            'mont': [3, 5, 6, 8, 10, 11, 23, 24],
            'ohm': [2, 3, 4, 5, 14, 16, 21, 25, 28]
        }, {
            'mont': [4, 7, 9, 12, 13, 14, 18, 21, 25],
            'ohm': [1, 8, 10, 19, 20, 22, 23, 29, 31]
        }, {
            'mont': [1, 2, 15, 16, 17, 19, 20, 22],
            'ohm': [6, 7, 11, 13, 18, 24, 27, 32]
        }
    ]

    meta = pd.read_csv('%s/RAW/label/IS2009/5-classes.csv' % base_path)

    school = list(map(lambda x: 'mont' if x[0:4] == 'Mont' else 'ohm', meta['filename'].values))
    speaker = list(map(lambda x: int(x.split('_')[1]), meta['filename'].values))

    def result_mapper(fold):
        res = []

        for i in range(len(meta['filename'].values)):
            if school[i] == 'mont':
                res.append(speaker[i] in fold['mont'])
            else:
                res.append(speaker[i] in fold['ohm'])

        return res

    data = np.load('%s/CORPORA/data/corpora.npy' % base_path)

    # Get index of each subset
    train, test, valid = list(map(result_mapper, folds))
    train, test, valid = data[train], data[test], data[valid]

    return train, test, valid


def load_fau(data_set=FAUDataSets.school, feature_name=FeatureSets.spectrogram, m_bands=40):
    """
    Load FAU-Aibo data set.

    :param data_set: data set name, type of FAUDataSets
    :type data_set: FAUDataSets

    :param feature_name: Feature used
    :type feature_name: FeatureSets

    :param m_bands: spectrogram resolution

    :return: Two Numpy array for data and label
    """
    print('==> Loading %s data...' % data_set.value)

    # Generate meta data
    if data_set == FAUDataSets.school:
        emotion_list = ['A', 'E', 'N', 'P', 'R']
        meta = pd.read_csv('%s/RAW/label/IS2009/5-classes.csv' % base_path)
        emotion = np.array(list(map(lambda x: emotion_list.index(x), meta['emotion'].values)))
    else:
        meta = pd.read_csv('%s/RAW/label/IS2009/2-classes.csv' % base_path)
        emotion = np.array(list(map(lambda x: 0 if x == 'NEG' else 1, meta['emotion'].values)))
    school = np.array(list(map(lambda x: 'mont' if x[0:4] == 'Mont' else 'ohm', meta['filename'].values)))

    # Generate data path
    data_path_map = {
        FeatureSets.spectrogram: '%s/SPECTROGRAMS/window-25-10/bands-%s/data.npy' % (base_path, m_bands),
        FeatureSets.egemaps: '%s/eGeMAPS/framed/framed.npy' % base_path,
        FeatureSets.compare: '%s/ComParE/framed/framed.npy' % base_path
    }

    # Load data
    corpora = np.load(data_path_map[feature_name])

    # Generate dim
    dim_map = {
        FAUDataSets.binary: 2,
        FAUDataSets.school: 5
    }

    train = [
        np.expand_dims(corpora[school == 'ohm'], axis=-1),
        to_categorical(emotion[school == 'ohm'], dim_map[data_set])
    ]

    tests = [
        np.expand_dims(corpora[school == 'mont'], axis=-1),
        to_categorical(emotion[school == 'mont'], dim_map[data_set])
    ]

    return train, tests
