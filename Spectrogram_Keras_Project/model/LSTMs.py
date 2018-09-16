from keras.models import Model
from keras.layers import Input, Dense, Bidirectional, CuDNNLSTM
from keras.layers import Dropout, Flatten, Activation, TimeDistributed, Reshape
from keras.layers import concatenate
from keras.layers import average


def lstm_final_pooling(previous_layer, layers, bidirectional=True):
    def wrapper(layer):
        if bidirectional:
            return Bidirectional(layer)
        else:
            return layer

    # Attention branch
    att = TimeDistributed(Flatten())(previous_layer)
    if layers != 1:
        att = wrapper(CuDNNLSTM(128, return_sequences=True))(att)
    else:
        att = wrapper(CuDNNLSTM(128, return_sequences=False))(att)

    for _ in list(range(1, layers)):
        att = Dropout(0.25)(att)
        if _ == layers - 1:
            att = wrapper(CuDNNLSTM(128, return_sequences=False))(att)
        else:
            att = wrapper(CuDNNLSTM(128, return_sequences=True))(att)
    print()
    att = Reshape((256, 1))(att)
    return att


def lstm_average_pooling(previous_layer, layers, bidirectional=True):
    def wrapper(layer):
        if bidirectional:
            return Bidirectional(layer)
        else:
            return layer

    # Attention branch
    att = TimeDistributed(Flatten())(previous_layer)
    if layers != 1:
        att = wrapper(CuDNNLSTM(128, return_sequences=True))(att)
    else:
        att = wrapper(CuDNNLSTM(128, return_sequences=False))(att)

    for _ in list(range(1, layers)):
        att = Dropout(0.25)(att)
        if _ == layers - 1:
            att = wrapper(CuDNNLSTM(128, return_sequences=False))(att)
        else:
            att = wrapper(CuDNNLSTM(128, return_sequences=True))(att)
    print()
    att = average(att)
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


def model_lstm_final_pooling(data_length, m_bands, output_size=4, blstm_layers=2, bidirectional=True):
    print('==> Building Model', data_length, m_bands)

    # Input
    inputs = Input(shape=(data_length, m_bands, 1))

    # Add GaussianNoise
    # fst = GaussianNoise(0.2)(inputs)
    fst = inputs

    lstm = lstm_final_pooling(fst, blstm_layers, bidirectional=bidirectional)

    # Concatenating
    # concat = merge([att, cnn], mode='mul')
    concat = Flatten(name='feature_concat')(lstm)

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


def model_lstm_average_pooling(data_length, m_bands, output_size=4, blstm_layers=2, bidirectional=True):
    print('==> Building Model', data_length, m_bands)

    # Input
    inputs = Input(shape=(data_length, m_bands, 1))

    # Add GaussianNoise
    # fst = GaussianNoise(0.2)(inputs)
    fst = inputs

    lstm = lstm_average_pooling(fst, blstm_layers, bidirectional=bidirectional)

    # Concatenating
    # concat = merge([att, cnn], mode='mul')
    concat = Flatten(name='feature_concat')(lstm)

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
