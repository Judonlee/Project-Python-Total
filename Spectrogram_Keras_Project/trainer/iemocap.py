import os

# Loaders
from datetime import datetime
from loader.iemocap import load_iemocap, IEMDataSets
from utils.model_utils import FeatureSets
from trainer.base import generate_callback_list, training_engine
from trainer.base import training_engine_final_pooling

####

def smile_trainer(feature_name='spectrogram', fold_range=range(10)):
    """
    Training for SMILE problem

    :param feature_name: which feature to train
    :param fold_range: range of 10 fold to be trained
    """
    archive_base = '/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/shared/EXP_ARCHIVE/IEM_SMILE'
    archive_base = archive_base + '/%s' % datetime.now().strftime("%Y-%m-%d-%H:%M")

    # Tweak this for different data
    data = load_iemocap(
        data_set=IEMDataSets.improvise,
        feature_set=FeatureSets.spectrogram,
        win_length=25,
        m_bands=40
    )

    for idx in fold_range:
        item = data[idx]
        train, tests, valid = item['data_train'], item['data_tests'], item['data_valid']

        base_path = '%s/%s/%s/' % (archive_base, feature_name, idx)
        os.system('mkdir -p %s' % base_path)

        callbacks_list = generate_callback_list(
            tests,
            valid,
            base_path
        )

        # Tweak this for different training task
        training_engine(
            train,
            tests,
            callbacks_list
        )


def smile_trainer_lstm_final_pooling(feature_name='spectrogram', fold_range=range(10)):
    """
    Training for SMILE problem

    :param feature_name: which feature to train
    :param fold_range: range of 10 fold to be trained
    """
    archive_base = '/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/shared/EXP_ARCHIVE/IEM_SMILE'
    archive_base = archive_base + '/%s' % datetime.now().strftime("%Y-%m-%d-%H:%M")

    # Tweak this for different data
    data = load_iemocap(
        data_set=IEMDataSets.improvise,
        feature_set=FeatureSets.spectrogram,
        win_length=25,
        m_bands=100
    )

    for idx in fold_range:
        item = data[idx]
        train, tests, valid = item['data_train'], item['data_tests'], item['data_valid']

        base_path = '%s/%s/%s/' % (archive_base, feature_name, idx)
        os.system('mkdir -p %s' % base_path)

        callbacks_list = generate_callback_list(
            tests,
            valid,
            base_path
        )

        # Tweak this for different training task
        training_engine_final_pooling(
            train,
            tests,
            callbacks_list
        )
