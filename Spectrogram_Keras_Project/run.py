'''
import sys
import os

#print

gpu = sys.argv[sys.argv.index('--gpu') + 1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu'''
from trainer.iemocap import smile_trainer_lstm_final_pooling

smile_trainer_lstm_final_pooling()
