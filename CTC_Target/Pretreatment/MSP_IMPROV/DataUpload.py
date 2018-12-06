import oss2
import os

auth = oss2.Auth('LTAI8i731E878sAl', 'w5esg3LdrPAiDSPKuDy6hsMdWGrrB0')
bucket = oss2.Bucket(auth, 'http://oss-cn-beijing.aliyuncs.com', 'voicestestbzt')

loadpath = 'D:/ProjectData/MSP-IMPROVE/Voice-Resample/'
for indexA in os.listdir(loadpath)[3:]:
    for indexB in os.listdir(os.path.join(loadpath, indexA)):
        for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
            for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                print(indexA, indexB, indexC, indexD)
                bucket.put_object_from_file('Treatment/%s/%s/%s/%s' % (indexA, indexB, indexC, indexD),
                                            os.path.join(loadpath, indexA, indexB, indexC, indexD))
