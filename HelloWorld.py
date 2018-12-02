import tensorflow as tf

import numpy as np

params = np.reshape(np.arange(1, 26, 1), [5, 5])
print(params)

ids = [4, 2, 3]

with tf.Session() as sess:
    print(sess.run(tf.nn.embedding_lookup(params, ids)))
