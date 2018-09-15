import csv
import pandas as pd
import numpy as np

from keras.models import Model
from keras.callbacks import Callback

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


def pretty_result(y, y_prd, info):
    eql = np.equal(y, y_prd)
    crt = y_prd[np.where(eql)]

    dst = np.bincount(y)
    prd = np.bincount(crt)
    d_len = dst.shape[0]

    # Bug fix
    # April 9, 2018
    # Some times model give no predict result on the last category
    # This will cause prd array has less length than dst
    prd = np.pad(prd, (0, d_len - prd.shape[0]), 'constant', constant_values=(0,0))
    
    wa = np.sum(prd) / np.sum(dst)
    ua = np.average(np.divide(prd, dst))

    print('---------------------------------------------------------')
    print('              %s' % info)
    print('=========================================================')

    print('- WA               : ', wa)
    print('- UA               : ', ua)
    print('- Y Data length    : ', dst)
    print('- Correctness      : ', prd)
    print('- Confusion Matrix :' )

    mtx = np.array([[ y[(y == i) & (y_prd == j)].shape[0] for j in range(d_len) ] for i in range(d_len) ])

    recall = np.array([mtx[i][i] / np.sum(mtx[i]) for i in range(mtx.shape[0]) ])
    precision = np.array([mtx[i][i] / np.sum(mtx.transpose()[i]) for i in range(mtx.shape[0]) ])

    recall = np.array(['%.1f%%' % (i * 100) for i in recall])
    recall = np.expand_dims(recall, axis=-1)
    precision = np.concatenate([precision, [0]])
    precision = np.array(['%.1f%%' % (i * 100) for i in precision])

    mtx = mtx.astype('str')

    mtx = np.hstack([mtx, recall])
    mtx = np.vstack([mtx, [precision]])

    col = [i for i in range(d_len)]
    idx = [i for i in range(d_len)]
    col.append('Recall')
    idx.append('Precision')

    df = pd.DataFrame(mtx, columns=col, index=idx)
    df.columns.names = ['Classified as ->']

    print(df)

    print('---------------------------------------------------------')
    
    return [ua, wa]


class SavePredictResult(Callback):
    def __init__(self, filepath, test_data, vald_data, batch_size):
        super().__init__()
        self.test_data = test_data
        self.vald_data = vald_data
        self.filepath = filepath
        self.batch_size = batch_size

        self.file = open('%s/hk_logs.csv' % filepath, 'w')
        self.csv = csv.writer(self.file, delimiter=',')
        self.csv.writerow(['Test WA', 'Test UA', 'Vald WA', 'Vald UA'])
        self.file.flush()

    def on_epoch_end(self, epoch, logs={}):
        # Test Data
        x, y = self.test_data
        res = self.model.predict(x, batch_size=self.batch_size)

        y = np.argmax(y, axis=-1)
        p = np.argmax(res, axis=-1)
        
        t_ua, t_wa = pretty_result(y, p, 'Result for test data')
        
        if self.vald_data is None:
            self.csv.writerow([t_wa, t_ua, '-', '-'])
            self.file.flush()
            return
        
        # Validation Data
        x, y = self.vald_data
        res = self.model.predict(x, batch_size=self.batch_size)
        
        # Saving stacked result
        stk = np.hstack([res, y])
        np.savetxt('%s/predict-%s.csv' % (self.filepath, epoch + 1), stk, delimiter=',')
        
        y = np.argmax(y, axis=-1)
        p = np.argmax(res, axis=-1)
        
        v_ua, v_wa = pretty_result(y, p, 'Result for validation data')
        
        self.csv.writerow([t_wa, t_ua, v_wa, v_ua])
        self.file.flush()


class SVMPredictor(Callback):
    def __init__(self, layer_before, train_data, vald_data):
        super().__init__()
        self.layer_before = layer_before
        self.vald_data = vald_data
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs={}):
        x_vald, y_vald = self.vald_data
        x_train, y_train = self.train_data
        
        y_valid = np.argmax(y_vald, axis=-1)
        y_train = np.argmax(y_train, axis=-1)
        
        extractor = Model(inputs=self.model.input,
                          outputs=self.model.get_layer(self.layer_before).output)

        ex_train = extractor.predict(x_train)
        ex_valid = extractor.predict(x_vald)
        
        scaler = StandardScaler()
        ex_train = scaler.fit_transform(ex_train)
        ex_valid = scaler.transform(ex_valid)

        sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, shuffle=True, n_jobs=-1)
        sgd.fit(ex_train, y_train)
        p_train = sgd.predict(ex_train)
        p_valid = sgd.predict(ex_valid)

        pretty_result(y_train, p_train, 'Result for SVM train')
        pretty_result(y_valid, p_valid, 'Result for SVM valid')
