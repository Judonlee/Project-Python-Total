import os

from utils.result_utils import SavePredictResult, SVMPredictor
from keras.callbacks import ModelCheckpoint, CSVLogger

# Model
from model.attention_blstm_fcn import model_attention_blstm_fcn
from model.LSTMs import model_lstm_final_pooling


def generate_callback_list(data_tests, data_valid, base_path, batch_size=32, svm_predictor=False):
    """
    Generate callback list

    :param data_tests: test data
    :param data_valid: validation data
    :param base_path: base_path for result saving
    :param batch_size: batch size

    :return: Keras callback list
    """
    x_tests, y_tests = data_tests
    x_valid, y_valid = data_valid

    weights_path = '%s/weights-{epoch:02d}.hdf5' % base_path
    os.system('mkdir -p %s' % base_path)

    callbacks_list = [
        ModelCheckpoint(
            weights_path,
            monitor='val_acc',
            verbose=1,
            save_best_only=True
        ),
        CSVLogger(
            '%s/log.csv' % base_path,
            separator=',',
            append=False
        ),
        SavePredictResult(
            base_path,
            test_data=(x_tests, y_tests),
            vald_data=(x_valid, y_valid),
            batch_size=batch_size
        )
    ]

    if svm_predictor:
        callbacks_list.append(SVMPredictor(
            layer_before='feature_concat',
            train_data=(x_tests, y_tests),
            vald_data=(x_valid, y_valid)
        ))

    return callbacks_list


def training_engine(data_train, data_tests, callbacks_list,
                    blstm_layers=2, with_fcn=True,
                    bidirectional=True, batch_size=32, epoch=100):
    """
    Generic training engine

    Designed with various parallel params to tweak training process.

    :param data_train: training data
    :param data_tests: test data
    :param callbacks_list: callbacks generated outsides
    :param blstm_layers: how many blstm layers are used
    :param with_fcn: if fcn is used
    :param bidirectional: if all blstm layers are bidirectional
    :param batch_size: batch size
    :param epoch: epoch
    """

    x_train, y_train = data_train
    x_tests, y_tests = data_tests

    model = model_attention_blstm_fcn(
        x_train.shape[-3],
        x_train.shape[-2],
        output_size=y_train.shape[-1]
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        validation_data=(x_tests, y_tests),
        shuffle=True,
        callbacks=callbacks_list
    )


def training_engine_final_pooling(data_train, data_tests, callbacks_list,
                                  blstm_layers=2, with_fcn=True,
                                  bidirectional=True, batch_size=32, epoch=100):
    """
    Generic training engine

    Designed with various parallel params to tweak training process.

    :param data_train: training data
    :param data_tests: test data
    :param callbacks_list: callbacks generated outsides
    :param blstm_layers: how many blstm layers are used
    :param with_fcn: if fcn is used
    :param bidirectional: if all blstm layers are bidirectional
    :param batch_size: batch size
    :param epoch: epoch
    """

    x_train, y_train = data_train
    x_tests, y_tests = data_tests

    model = model_lstm_final_pooling(x_train.shape[-3], x_train.shape[-2], output_size=y_train.shape[-1],
                                     blstm_layers=blstm_layers, bidirectional=bidirectional)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        validation_data=(x_tests, y_tests),
        shuffle=True,
        callbacks=callbacks_list
    )
