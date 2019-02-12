from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.DBLSTM import DBLSTM
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer

if __name__ == '__main__':
    trainData, trainSeq, trainLabel, testData, testSeq, testLabel = Load_DBLSTM()
    classifier = DBLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                        firstAttention=StandardAttentionInitializer, secondAttention=None,
                        firstAttentionShape=None, secondAttentionShape=None,
                        firstAttentionName='FirstAttention', secondAttentionName=None)
    classifier.Valid()
