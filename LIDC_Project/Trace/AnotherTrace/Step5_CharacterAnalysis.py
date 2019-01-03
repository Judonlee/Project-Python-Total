import os

if __name__ == '__main__':
    characterPath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition/'

    for indexA in os.listdir(characterPath):
        for indexB in os.listdir(os.path.join(characterPath, indexA)):
            with open(os.path.join(characterPath, indexA, indexB, 'Character.txt'), 'r') as file:
                text = file.read()
                # print(text)
                flag1 = text[text.find('<internalStructure>') + len('<internalStructure>'):text.find(
                    '</internalStructure>')] == '1'
                flag2 = text[text.find('<calcification>') + len('<calcification>'):text.find(
                    '</calcification>')] == '6'
                flag3 = text[text.find('<texture>') + len('<texture>'):text.find('</texture>')] == '5'
                print(indexA, indexB, flag1, flag2, flag3)

            with open(os.path.join(characterPath, indexA, indexB, 'CharacterDecision.txt'), 'w') as file:
                file.write(str(int(flag1)) + ',' + str(int(flag2)) + ',' + str(int(flag3)))
            # exit()
