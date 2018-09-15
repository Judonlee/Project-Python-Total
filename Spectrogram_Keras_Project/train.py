import sys
import os

#print

gpu = sys.argv[sys.argv.index('--gpu') + 1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import tensorflow as tf

# Tensorflow session tweak
config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# -------------------------------------------------------
# DO NOT EDIT ABOVE

# Business code
# from trainer.iemocap import smile_trainer

data_set = sys.argv[sys.argv.index('--data-set') + 1]

# Training for ComParE feature set on FAU-Aibo
from trainer.fau import smile_trainer
smile_trainer(data_set=data_set, feature_set='compare')
