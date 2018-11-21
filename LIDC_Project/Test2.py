import os

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step5-NodulesCsv/'
    counter = 0
    for fold in os.listdir(loadpath):
        for pic in os.listdir(os.path.join(loadpath, fold)):
            counter+=len(os.listdir(os.path.join(loadpath, fold, pic)))
    print(counter)
