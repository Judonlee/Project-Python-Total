from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Features/GeMAPSv01a-Npy/Appoint-0/'

    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
        IEMOCAP_Loader_Npy(loadpath=loadpath)

