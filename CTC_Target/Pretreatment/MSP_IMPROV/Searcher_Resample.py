import os
import librosa

if __name__ == '__main__':
    voicepath = 'D:/ProjectData/MSP-IMPROVE/Voice/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/Voice-Resample/'
    for indexA in os.listdir(voicepath):
        # os.makedirs(os.path.join(savepath, indexA))
        for indexB in os.listdir(os.path.join(voicepath, indexA)):
            for indexC in os.listdir(os.path.join(voicepath, indexA, indexB)):
                os.makedirs(os.path.join(savepath, indexA, indexB, indexC))
                for indexD in os.listdir(os.path.join(voicepath, indexA, indexB, indexC)):
                    print(indexA, indexB, indexC, indexD)
                    data, sr = librosa.load(os.path.join(voicepath, indexA, indexB, indexC, indexD), sr=16000)
                    librosa.output.write_wav(path=os.path.join(savepath, indexA, indexB, indexC, indexD), y=data, sr=sr)
