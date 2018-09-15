from keras.models import Model
from keras.layers import Input, LSTM, Input, Dense, Bidirectional, GaussianNoise, CuDNNLSTM
from keras.layers import Conv1D, Conv2D
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalAveragePooling1D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization, Flatten, Activation, TimeDistributed, Reshape
from keras.layers import concatenate, merge

from utils.layer_ext import Attention


def branch_fcn(previous_layer):
    """
    FCN Branch generation

    :param previous_layer: Layer before FCN
    :return: FCN layer stack
    """
    fcn = Conv2D(64, (8, 8))(previous_layer)
    fcn = MaxPooling2D(pool_size=(2, 2))(fcn)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dropout(0.25)(fcn)

    fcn = Conv2D(128, (5, 5))(fcn)
    fcn = MaxPooling2D(pool_size=(2, 2))(fcn)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dropout(0.25)(fcn)

    fcn = Conv2D(128, (3, 3))(fcn)
    # fcn = MaxPooling2D(pool_size=(2, 2))(fcn)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dropout(0.25)(fcn)

    fcn = GlobalMaxPooling2D()(fcn)

    fcn = Reshape((128, 1))(fcn)

    return fcn


def branch_attention(previous_layer, layers, bidirectional=True):
    """
    Attention BLSTM branch generation

    :param previous_layer: Layer before attention blstm branch
    :param layers: How many blstm layers
    :param bidirectional: If blstm layers are bidirectional
    :return: Attention BLSTM branch stack
    """

    def wrapper(layer):
        if bidirectional:
            return Bidirectional(layer)
        else:
            return layer

    # Attention branch
    att = TimeDistributed(Flatten())(previous_layer)
    att = wrapper(CuDNNLSTM(128, return_sequences=True))(att)

    for _ in list(range(1, layers)):
        att = Dropout(0.25)(att)
        att = wrapper(CuDNNLSTM(128, return_sequences=True))(att)

    att = Attention()(att)

    # att = Dense(feature_length, activation='relu')(att)
    att = Reshape((256, 1))(att)
    return att


def section_dnn(previous_layer):
    """
    DNN section generation

    You may rewrite this function for further research

    :param previous_layer: Layer before DNN
    :return: DNN section layers
    """
    # Classifier
    clf = Dense(256)(previous_layer)
    clf = Activation('relu')(clf)
    # clf = Dropout(0.2)(clf)
    return clf


def model_attention_blstm_fcn(data_length, m_bands, output_size=4,
                              blstm_layers=2, with_fcn=True, bidirectional=True):
    """
    Attention BLSTM FCN model

    model generation with following params.

    :param data_length: data length
    :param m_bands: spectrogram resolution or feature dimension
    :param output_size: DNN output size
    :param blstm_layers: How many blstm layers
    :param with_fcn: If FCN engaged
    :param bidirectional: If blstm layers are bidirectional
    :return: Keras model for Attention BLSTM FCN
    """
    print('==> Building Model', data_length, m_bands)

    # Input
    inputs = Input(shape=(data_length, m_bands, 1))

    # Add GaussianNoise
    # fst = GaussianNoise(0.2)(inputs)
    fst = inputs

    att = branch_attention(fst, blstm_layers, bidirectional=bidirectional)

    # CNN branch
    if with_fcn:
        fcn = branch_fcn(fst)
        concat = concatenate([att, fcn], axis=1)
    else:
        concat = concatenate([att], axis=1)

    # Concatenating
    # concat = merge([att, cnn], mode='mul')
    concat = Flatten(name='feature_concat')(concat)

    dnn = section_dnn(concat)

    # Out
    outputs = Dense(output_size, activation='softmax')(dnn)

    # RMSprop(lr=0.0004)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    model.summary()

    return model
