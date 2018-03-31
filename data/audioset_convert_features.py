import os
import tensorflow as tf
import numpy as np

tfrecord_path = "audioset/audioset_v1_embeddings/bal_train/zy.tfrecord"

for item in tf.python_io.tf_record_iterator(tfrecord_path):
    tf_item = tf.train.Example.FromString(item)
    if "audio_embedding" in tf_item.features.feature:
        print(tf_item)
        break
