from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader

if __name__ == '__main__':
    trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
        loadpath='D:/ProjectData/Project-CTC-Data/Transcription-SingleNumber-Class5/', appoint=0)
    print(trainScription)
