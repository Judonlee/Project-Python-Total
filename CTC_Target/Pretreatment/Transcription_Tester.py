import os

if __name__ == '__main__':
    with open(r'E:\LaboratoryData-Origin\FAU-AEC\transliteration\lexicon.txt', 'r') as file:
        data = file.readlines()
    dictionary = {}
    for sample in data:
        dictionary[sample.split('\t')[0]] = sample.split('\t')[1][:-1]
    # for key in dictionary.keys():
    #     print(key, '\t', dictionary[key])

    loadpath = 'D:/ProjectData/FAU-AEC-Treated/Transcription/'
    savepath = 'D:/ProjectData/FAU-AEC-Treated/Transcription-Pronouncing/'
    if not os.path.exists(savepath): os.makedirs(savepath)

    for filename in os.listdir(loadpath):
        print(filename)
        with open(os.path.join(loadpath, filename), 'r') as file:
            data = file.read()
        transcription = ''
        for sample in data.split(' '):
            if sample in dictionary.keys():
                transcription += dictionary[sample].replace('|', ' ').replace('\'', '') + ' '
        # print('Origin :', data, '\nPronouncing :', transcription, '\n')
        with open(os.path.join(savepath, filename), 'w') as file:
            file.write(transcription)
