import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/GitHub/CTC_Project_Again/TrainRestart/Tester/Result_Double_BLSTM_CTC_CRF_%s/Bands-%d-Session-%d-%s-%s/%04d.csv'
    bands = 30
    for testgender in ['Female', 'Male']:
        for session in range(1, 6):
            uaList, waList = [0], [0]

            for part in ['UA', 'WA']:
                for traingender in [testgender]:
                    for episode in range(100):
                        if not os.path.exists(
                                loadpath % (part, bands, session, traingender, testgender, episode)): continue
                        data = numpy.genfromtxt(
                            fname=loadpath % (part, bands, session, traingender, testgender, episode), dtype=float,
                            delimiter=',')
                        WA, UA = 0, 0
                        for index in range(len(data)):
                            WA += data[index][index]
                            UA += data[index][index] / sum(data[index])
                        WA = WA / sum(sum(data))
                        UA = UA / len(data)

                        waList.append(WA)
                        uaList.append(UA)

            print(numpy.average(waList))
