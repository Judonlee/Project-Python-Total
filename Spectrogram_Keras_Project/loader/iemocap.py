import numpy as np
from enum import Enum

from keras.utils import to_categorical
from utils.model_utils import FeatureSets


class IEMDataItem(object):
    def __init__(self, data_set, gender, session):
        """
        Constructor

        :param data_set: which data set
        :type data_set: IEMDataSets

        :param gender: which gender
        :param session: which session
        """
        self.data_set = data_set
        self.gender = gender
        self.session = session

    def dist(self, prefix=''):
        """
        Generate final file path

        :param prefix: file prefix, typical for path
        :return: string for file path
        """
        return '%s%s-%s-%s.npy' % (prefix, self.data_set.value, self.gender, self.session)


class IEMDataSets(Enum):
    improvise = 'improvise'
    script = 'script'
    corpora = 'corpora'


def generate_iemocap_item():
    """
    IEMOCAP base loader

    Loading base data from disk, generating basic object,
    generate_iemocap_item does not read data from disk, only generate
    data item object.

    :return: A numpy array represents the data file name.
    """
    sessions = [1, 2, 3, 4, 5]
    genders = ['female', 'male']
    data_sets = [IEMDataSets.improvise, IEMDataSets.script]

    d_list = [[[IEMDataItem(t, g, s) for s in sessions] for g in genders] for t in data_sets]
    d_list = np.array(d_list).flatten()

    return d_list


def adapt_corpora(data):
    """
    Adapter for corpora data items

    :param data: data list
    :return: Adapted data
    """
    data = data.reshape(2, 10)
    data = [np.concatenate([data[0][i], data[1][i]]) for i in range(10)]
    data = np.array(data)
    return data


def load_iemocap(data_set, feature_set, win_length=25, m_bands=40, with_valid=True):
    """
    Load IEMOCAP data set

    :param data_set: data set name, type of IEMDataSets
    :type data_set: IEMDataSets

    :param feature_set: feature name, type of FeatureSets
    :type feature_set: FeatureSets

    :param win_length: hamming window length
    :param m_bands: spectrogram resolution
    :param with_valid: if validation data is generated

    :return: Numpy array for test, validation, training data
    """
    cnd = not feature_set == FeatureSets.spectrogram
    cnd = cnd and not win_length == 25
    cnd = cnd and not m_bands == 40
    if cnd:
        print('==> Warning feature sets except spectrogram does not include different win_length and m_bands variation')

    base_path = '/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/shared/IEMOCAP'
    path_dict = {
        FeatureSets.spectrogram: '%s/MERGED_SPECTROGRAMS/window-%s-10/bands-%s/' % (base_path, win_length, m_bands),
        FeatureSets.egemaps: '%s/MERGED_eGeMAPS/framed/' % base_path,
        FeatureSets.compare: '%s/MERGED_ComParE/framed/' % base_path
    }

    path_prefix = path_dict[feature_set]
    file_list = generate_iemocap_item()

    file_list = filter(lambda x: x.data_set == data_set or data_set == IEMDataSets.corpora, file_list)
    file_list = list(file_list)

    data_list = map(lambda x: np.load(x.dist(prefix=path_prefix + 'data/')), file_list)
    data_list = list(data_list)
    data_list = np.array(data_list)

    label_list = map(lambda x: np.load(x.dist(prefix=path_prefix + 'label/')), file_list)
    label_list = list(label_list)
    label_list = np.array(label_list)

    # Adapt corpora data set
    if data_set == IEMDataSets.corpora:
        data_list, label_list = adapt_corpora(data_list), adapt_corpora(label_list)

    # Split test, validation, training data
    d_10_fold = []

    for t_idx in range(10):
        # value for valid data
        v_idx = 1 if t_idx % 2 == 0 else -1
        v_idx = t_idx + v_idx

        if with_valid:
            d_10_fold.append({
                'data_valid': [
                    np.expand_dims(data_list[v_idx], axis=-1),
                    to_categorical(label_list[v_idx], 4)
                ],
                'data_tests': [
                    np.expand_dims(data_list[t_idx], axis=-1),
                    to_categorical(label_list[t_idx], 4)
                ],
                'data_train': [
                    np.expand_dims(np.concatenate([data_list[k] for k in range(10) if k != t_idx and k != v_idx]),
                                   axis=-1),
                    to_categorical(np.concatenate([label_list[k] for k in range(10) if k != t_idx and k != v_idx]), 4)
                ]
            })
        else:
            d_10_fold.append({
                'data_tests': [
                    np.expand_dims(data_list[t_idx], axis=-1),
                    to_categorical(label_list[t_idx], 4)
                ],
                'data_train': [
                    np.expand_dims(np.concatenate([data_list[k] for k in range(10) if k != t_idx]), axis=-1),
                    to_categorical(np.concatenate([label_list[k] for k in range(10) if k != t_idx]), 4)
                ]
            })

    return d_10_fold
