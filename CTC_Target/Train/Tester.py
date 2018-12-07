from CTC_Target.Loader.IEMOCAP_Loader import Load, LoadSpecialLabel
import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import os

if __name__ == '__main__':
    transcriptionPath = 'E:/CTC_Target/Features/PronouncingDictionaryDouble/'
    for part in ['Bands30', 'Bands40']:
        loadpath = 'E:/CTC_Target/Features/%s/' % part
        session = 0
        savepath = 'CTC-Origin_PDW/%s-Session-%d/' % (part, session)
        if os.path.exists(savepath): continue
        os.makedirs(savepath)
        trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = LoadSpecialLabel(
            loadpath=loadpath, appoint=session, transcriptionpath=transcriptionPath)
        for index in range(len(trainData)):
            print(len(trainData[index]), len(trainScription[index]))
            if len(trainData[index]) < len(trainScription[index]):
                print('ERROR')
                exit()
