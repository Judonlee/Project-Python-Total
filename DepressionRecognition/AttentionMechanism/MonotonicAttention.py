import tensorflow
from tensorflow.contrib import signal
import numpy


def MonotonicAttentionInitializer(dataInput, scopeName, hiddenNoduleNumber, attentionScope=None, blstmFlag=True):
    def MovingSum(tensor, backward, forward, namescope):
        with tensorflow.name_scope(namescope):
            networkParameter['%s_Pad' % namescope] = tensorflow.pad(tensor, [[0, 0], [backward, forward]],
                                                                    name='%s_Pad' % namescope)
            networkParameter['%s_Expand' % namescope] = tensorflow.expand_dims(
                input=networkParameter['%s_Pad' % namescope], axis=-1, name='%s_Expand' % namescope)
            networkParameter['%s_Filter' % namescope] = tensorflow.ones(
                shape=[backward + forward + 1, 1, 1], dtype=tensorflow.float32, name='%s_Filter' % namescope)
            networkParameter['%s_Sum' % namescope] = tensorflow.nn.conv1d(
                value=networkParameter['%s_Expand' % namescope], filters=networkParameter['%s_Filter' % namescope],
                stride=1, padding='VALID', name='%s_Sum' % namescope)
            networkParameter['%s_Result' % namescope] = networkParameter['%s_Sum' % namescope][..., 0]

    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput

        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['AttentionWeight_Denominator_Raw'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.exp,
            name='%s_AttentionWeight_Denominator_Raw' % scopeName)
        networkParameter['AttentionWeight_Denominator'] = tensorflow.maximum(tensorflow.maximum(
            x=networkParameter['AttentionWeight_Denominator_Raw'], y=1E-5, name='AttentionWeight_Denominator')[..., 0],
                                                                             1E-5)

        MovingSum(tensor=networkParameter['AttentionWeight_Denominator'], backward=attentionScope - 1, forward=0,
                  namescope='Denominator')

        networkParameter['AttentionWeight_Numerator'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.nn.tanh,
            name='%s_AttentionWeight_Numerator' % scopeName)[..., 0]

        networkParameter['AttentionWeight_Final'] = tensorflow.divide(
            x=networkParameter['AttentionWeight_Numerator'],
            y=tensorflow.maximum(networkParameter['AttentionWeight_Denominator'], 1E-5),
            name='AttentionWeight_Final')
        MovingSum(tensor=networkParameter['AttentionWeight_Final'], backward=0, forward=attentionScope - 1,
                  namescope='Probability')

        networkParameter['Probability_Supplement'] = tensorflow.tile(
            input=networkParameter['Probability_Result'][:, :, tensorflow.newaxis],
            multiples=[1, 1, hiddenNoduleNumber], name='Probability_Supplement')
        networkParameter['FinalResult_Media'] = tensorflow.multiply(x=networkParameter['DataInput'],
                                                                    y=networkParameter['Probability_Supplement'],
                                                                    name='FinalResult_Media')
        networkParameter['FinalResult'] = tensorflow.reduce_mean(input_tensor=networkParameter['FinalResult_Media'],
                                                                 axis=1, name='FinalResult')

    return networkParameter


def MonotonicChunkwiseAttentionInitializer(dataInput, scopeName, hiddenNoduleNumber, attentionScope=None,
                                           blstmFlag=True):
    def MovingMax(tensor, windowLen, namescope):
        with tensorflow.name_scope(namescope):
            networkParameter['%s_Pad' % namescope] = tensorflow.pad(tensor, paddings=[[0, 0], [windowLen - 1, 0]],
                                                                    name='%s_Pad' % namescope)
            networkParameter['%s_Reshape' % namescope] = tensorflow.reshape(
                tensor=networkParameter['%s_Pad' % namescope],
                shape=[tensorflow.shape(networkParameter['%s_Pad' % namescope])[0], 1,
                       tensorflow.shape(networkParameter['%s_Pad' % namescope])[1], 1],
                name='%s_Reshape' % namescope)
            networkParameter['%s_MaxPool' % namescope] = tensorflow.nn.max_pool(
                networkParameter['%s_Reshape' % namescope], ksize=[1, 1, windowLen, 1], strides=[1, 1, 1, 1],
                padding='VALID', name='%s_MaxPool' % namescope)[:, 0, :, 0]

    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput

        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['DataLogits'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.nn.tanh,
            name='DataLogits_%s' % scopeName)[..., 0]
        MovingMax(tensor=networkParameter['DataLogits'], windowLen=attentionScope, namescope='Logits_Max')

        #########################################################################

        networkParameter['DataLogits_Pad'] = tensorflow.pad(networkParameter['DataLogits'],
                                                            paddings=[[0, 0], [attentionScope - 1, 0]],
                                                            constant_values=0, name='DataLogits_Pad')
        networkParameter['DataLogits_Frame'] = signal.frame(
            signal=networkParameter['DataLogits_Pad'], frame_length=attentionScope, frame_step=1,
            name='DataLogits_Frame')
        networkParameter['DataLogits_Frame_Reduce'] = tensorflow.subtract(
            x=networkParameter['DataLogits_Frame'], y=networkParameter['Logits_Max_MaxPool'][:, :, tensorflow.newaxis],
            name='DataLogits_Frame_Reduce')
        networkParameter['Denominator_Softmax'] = tensorflow.reduce_sum(
            input_tensor=tensorflow.exp(x=networkParameter['DataLogits_Frame_Reduce']), axis=2,
            name='Denominator_Softmax')
        networkParameter['Denominator_Frame'] = signal.frame(signal=networkParameter['Denominator_Softmax'],
                                                             frame_length=attentionScope, frame_step=1, pad_end=True,
                                                             pad_value=0, name='Denominator_Frame')

        ########################################################################

        networkParameter['FrameMax'] = signal.frame(signal=networkParameter['Logits_Max_MaxPool'],
                                                    frame_length=attentionScope, frame_step=1, pad_end=True,
                                                    pad_value=0, name='FrameMax')
        networkParameter['DataLogits_Numerators'] = tensorflow.subtract(
            x=networkParameter['DataLogits'][:, :, tensorflow.newaxis], y=networkParameter['FrameMax'],
            name='DataLogits_Numerators')
        networkParameter['Numerators_Softmax'] = tensorflow.exp(x=networkParameter['DataLogits_Numerators'],
                                                                name='Numerators_Softmax')
        networkParameter['AttentionProbability_Raw'] = tensorflow.divide(
            x=networkParameter['Numerators_Softmax'], y=tensorflow.maximum(networkParameter['Denominator_Frame'], 1E-5),
            name='AttentionProbability_Raw')
        networkParameter['AttentionProbability'] = tensorflow.reduce_sum(
            input_tensor=networkParameter['AttentionProbability_Raw'], axis=2, name='AttentionProbability')

        networkParameter['AttentionProbability_Supplement'] = tensorflow.tile(
            input=networkParameter['AttentionProbability'][:, :, tensorflow.newaxis],
            multiples=[1, 1, hiddenNoduleNumber], name='AttentionProbability_Supplement')
        networkParameter['FinalResult_Media'] = tensorflow.multiply(
            x=networkParameter['AttentionProbability_Supplement'], y=networkParameter['DataInput'],
            name='FinalResult_Media')
        networkParameter['FinalResult'] = tensorflow.reduce_sum(input_tensor=networkParameter['FinalResult_Media'],
                                                                axis=1, name='FinalResult')

        return networkParameter
