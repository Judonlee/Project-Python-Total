import os

# Loaders
from loader.fau import load_fau, FAUDataSets

from datetime import datetime
from utils.model_utils import FeatureSets
from trainer.base import generate_callback_list, training_engine


def basic_trainer(m_bands=40, data_set='school', blstm_layers=2, with_fcn=True, bidirectional=True,):
    """
    Training for Spectrogram resolution problem

    :param m_bands: spectrogram resolution
    :param data_set: data set name
    :param blstm_layers: how many blstm layers
    :param with_fcn: if FCN branch is engaged
    :param bidirectional: if lstm layers are bidirectional
    """
    archive_base = '/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/shared/EXP_ARCHIVE/FAU_BASIC'
    archive_base = archive_base + '/%s' % datetime.now().strftime("%Y-%m-%d-%H:%M")

    train, tests = load_fau(
        data_set=FAUDataSets[data_set],
        feature_name=FeatureSets.spectrogram,
        m_bands=m_bands
    )

    # Path definition
    base_path = '%s/bands-%s/%s' % (archive_base, m_bands, data_set)
    os.system('mkdir -p %s' % base_path)

    callbacks_list = generate_callback_list(
        train,
        tests,
        base_path
    )

    print('==> Training')
    training_engine(
        train,
        tests,
        callbacks_list,
        blstm_layers=blstm_layers,
        with_fcn=with_fcn,
        bidirectional=bidirectional
    )


def smile_trainer(data_set='school', feature_set='spectrogram'):
    """
    Load FAU-Aibo smile feature set.

    :param data_set: data set
    :param feature_set: feature set
    """
    archive_base = '/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/shared/EXP_ARCHIVE/FAU_SMILE'
    archive_base = archive_base + '/%s' % datetime.now().strftime("%Y-%m-%d-%H:%M")

    train, tests = load_fau(
        data_set=FAUDataSets[data_set],
        feature_name=FeatureSets[feature_set]
    )

    # Path definition
    base_path = '%s/%s/%s' % (archive_base, feature_set, data_set)
    os.system('mkdir -p %s' % base_path)

    callbacks_list = generate_callback_list(
        train,
        tests,
        base_path
    )

    print('==> Training')
    training_engine(
        train,
        tests,
        callbacks_list
    )
