
# coding: utf-8

# # Spectrogram Extraction
# 
# - window size  [15ms, 25ms, 50ms, 100ms, 200ms]
# - window shift 10ms
# - mel bands    [30, 40, 60, 80, 100, 120]
# - data shape   (None, 500, m_bands)

# In[1]:


import pandas as pd
import numpy as np
import librosa
import ipywidgets as widgets
from scipy import signal

from multiprocessing import Pool


# In[2]:


def iemocap_param_generator(args):
    meta_file, data_session, data_gender, data_type = args
    
    meta = pd.read_csv(meta_file)
    
    conds = []
    conds_mark = np.array([[[[t, g, s] for s in data_session] for g in data_gender] for t in data_type])
    
    for t in data_type:
        for g in data_gender:
            for s in data_session:
                d = (meta.session == s) & (meta.gender == g) & (meta.type == t)
                conds.append(d)
    
    conds = np.array(conds)
    
    return meta, conds_mark, conds


# In[3]:


def iemocap_task_operator(argv):
    # Inject params
    data_filename, data_folder, data_session, data_emotion = argv
    global win_length, hop_length, m_bands
    
    # meta params
    emo_map = { 'neu': 0, 'hap': 1, 'exc': 1, 'sad': 2, 'ang': 3 }

    base_path = '/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/Hawkins/IEMOCAP/audio-set'
    
    seq_length = 500

    n_fft = win_length    
    
    # ==================
    # librosa processing
    wav_path = '%s/Session%s/wav/%s/%s.wav' % (base_path, data_session, data_folder, data_filename)
    
    # load data
    y, sr = librosa.load(wav_path, sr=None)
    # resampe data
    y = librosa.resample(y, sr, s_rate)
    
    # STFT and spectrogram
    D = np.abs(librosa.stft(y, 
                            n_fft=n_fft, 
                            win_length=win_length, 
                            hop_length=hop_length,
                            window=signal.hamming,
                            center=False)) ** 2

    S = librosa.feature.melspectrogram(S=D, n_mels=m_bands)
    gram = librosa.power_to_db(S, ref=np.max)
    
    # Zero padding
    t = gram.transpose().flatten()
    if t.shape[0] >= seq_length * m_bands:
        t_pad = t[0:seq_length * m_bands]
    else:
        length = seq_length * m_bands - t.shape[0]
        t_pad = np.pad(t, (0, length), 'constant', constant_values=(0, 0))
    
    return emo_map[data_emotion], t_pad.reshape(seq_length, m_bands)


# In[4]:


def task_assembler(meta_file):
    global w
    pool = Pool(32) # Equal to CPU cores
    
    meta, conds_mark, conds = iemocap_param_generator([
        meta_file, 
        [1, 2, 3, 4, 5],
        ['female', 'male'],
        ['improvise', 'script']
    ])
    
    conds_mark_shape = np.product(conds_mark.shape) // conds.shape[0]
    conds_mark = conds_mark.reshape((conds.shape[0], conds_mark_shape))
    
    # import ipdb; ipdb.set_trace()
    
    for i in range(conds.shape[0]):
        mark = conds_mark[i]

        print(mark)
        
        data_filename = meta[conds[i]].filename
        data_folder   = meta[conds[i]].folder
        data_session  = meta[conds[i]].session
        data_emotion  = meta[conds[i]].emotion


        data_params = np.array([data_filename, data_folder, data_session, data_emotion]).transpose()
        # import ipdb; ipdb.set_trace()
        
        data_gen = pool.map(iemocap_task_operator, data_params)
        
        data_gen_emo   = np.array([item[0] for item in data_gen])
        data_gen_value = np.array([item[1] for item in data_gen])
        
        save_path = './dist/IEMOCAP'
        save_name = '-'.join([str(i) for i in conds_mark[i]])
        
        np.save('%s/data/%s' % (save_path, save_name), data_gen_value)
        np.save('%s/label/%s' % (save_path, save_name), data_gen_emo)


# In[5]:


import shutil
# 0.015, 0.025, 0.050, 0.100, 
win_length_choice = [0.200]
m_bands_choice = [30, 40, 60, 80, 100, 120]

s_rate = 16000
win_length = int(0.025 * s_rate) # Window length 15ms, 25ms, 50ms, 100ms, 200ms
hop_length = int(0.010 * s_rate) # Window shift  10ms
m_bands = 40

for w in win_length_choice:
    for m in m_bands_choice:    
        m_bands = m
        win_length = int(w * s_rate)

        print('Task begin', (win_length / 16), 'ms', m_bands, 'mels')
        task_assembler('./meta/IEMOCAP/IEMOCAP.csv') # IEMOCAP_MERGE, IEMOCAP
        
        dir_base = '/mnt/a2f3b4e1-c182-4f8c-8eb2-5044d9b4ef28/shared/IEMOCAP_SPECTROGRAMS'
        dir_to = '%s/window-%s-10/bands-%s' % (dir_base, int(w * 1000), m)
        dir_from = './dist/IEMOCAP'
        
        shutil.copytree(dir_from, dir_to)


# In[ ]:


print('finished!')

