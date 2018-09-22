from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader

if __name__ == '__main__':
    trainData, trainLabel, testData, testLabel = IEMOCAP_Loader(loadpath='F:\\Project-CTC-Data\\Npy\\Bands30\\',
                                                                appoint=0)
