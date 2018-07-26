import tensorflow as tf
import numpy as np

data = np.arange(1, 100 + 1)
def create_batch(data,batch_size):
    data_input = tf.constant(data)

    batch_shuffle = tf.train.shuffle_batch([data_input], enqueue_many=True, batch_size=batch_size, capacity=100, min_after_dequeue=10, allow_smaller_final_batch=True)
    # batch_no_shuffle = tf.train.batch([data_input], enqueue_many=True, batch_size=10, capacity=100, allow_smaller_final_batch=True)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(100):
            print(i, sess.run([batch_shuffle]))
        coord.request_stop()
        coord.join(threads)

create_batch(data,10)