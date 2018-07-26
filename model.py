import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import json
from load_data import dataset,test_data
from func import config_data

def read_dict():
    with open("dict.json",encoding="utf-8") as data_file:
        data = json.load(data_file)
    return {data[i]:i for i in data}


def create_index_batch(data,batch_size):
    L=len(data)-1
    index=np.arange(L)
    data_input = tf.constant(index)
    batch_shuffle = tf.train.shuffle_batch([data_input], enqueue_many=True, batch_size=batch_size, capacity=L, min_after_dequeue=50, allow_smaller_final_batch=True)
    # batch_no_shuffle = tf.train.batch([data_input], enqueue_many=True, batch_size=10, capacity=100, allow_smaller_final_batch=True)

    # with tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in range(100):
    #         print(i,"*****", sess.run([batch_shuffle]),"*******")
    #     coord.request_stop()
    #     coord.join(threads)
    return batch_shuffle

def create_batch(data,index_batch):
    return [data[X] for X in index_batch]


# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(10000):
#         print(i,"*****", create_batch(data,sess.run(M)),"*******")
#     coord.request_stop()
#     coord.join(threads)

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.001
training_iters = 2000000
batch_size = 300
val_iters = 1
val_batch_size = 128
val_set_size = 5000
display_step = 10

# Network Parameters
seq_max_len = 250  # Sequence max length
n_hidden = 300  # hidden layer num of features
n_classes = 1109  # linear sequence or not
input_neural=6   # dim of feature sequence

data=dataset()
data=config_data(data,seq_max_len)
M=create_index_batch(data,batch_size)

# placeholder input
X=tf.placeholder("float",[None,seq_max_len,input_neural])
Y=tf.placeholder("float",[None,n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]),name="weights")
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]),name="biases")
}

def LSTM(X,seqlen,weights,biases):
    X=tf.unstack(X,seq_max_len,1)
    # lstm=rnn.BasicLSTMCell(n_hidden,name="lstm")
    # outputs,state=rnn.static_rnn(lstm,X,dtype=tf.float32,sequence_length=seqlen)



    lstm_F=rnn.BasicLSTMCell(n_hidden,name="lstm_F")

    lstm_B = rnn.BasicLSTMCell(n_hidden,name="lstm_B")

    # outputs, _, _ = rnn.static_bidirectional_rnn(lstm_F, lstm_B,X,
    #                                              dtype=tf.float32,sequence_length=seqlen)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_F, lstm_B, X,
                                              dtype=tf.float32,sequence_length=seqlen)


    outputs=tf.stack(outputs)
    outputs=tf.transpose(outputs,[1,0,2])
    print(outputs)
    batchsize = tf.shape(outputs)[0]
    index = tf.range(0, batchsize) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    print(outputs)
    return tf.matmul(outputs, weights['out']) + biases['out']


pred = LSTM(X, seqlen, weights, biases)

# Define loss and optimizer
out=tf.nn.softmax(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Evaluate model.ckpt
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    # config=tf.ConfigProto(log_device_placement=True)
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    step = 1
    while step * batch_size < training_iters:
        T=create_batch(data, sess.run(M))
        x=[x[0] for x in T]
        y=[y[1] for y in T]
        s=[s[2]  for s in T]
        batch_x, batch_y, batch_seqlen = x,y,s
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y,
                                       seqlen: batch_seqlen})

        step += 1
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={X: batch_x, Y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            if loss<0.05:
                break
    save_path = saver.save(sess, "model_2/")

    # saver.restore(sess, "/tmp/model.ckpt.ckpt")
    print("Model restored.")
    Dict=read_dict()
    while  True:
        try:
            test = json.loads(input("input: "))
            test=test["requests"][0]["ink"]
            data = test_data(test,"入")
            Dtest=config_data([data],seq_max_len)
            x = [x[0] for x in Dtest]
            y = [y[1] for y in Dtest]
            s = [s[2] for s in Dtest]

            print(Dict[np.array(sess.run(out,feed_dict={X:x,Y:y,seqlen:s})).argmax()+1])
            # print(out)
            # for i in range(len(out)):
            #     if out[i]==max(out):
            #         print(i)
            tim = time.time()
            Output = list(sess.run(out, feed_dict={X: x, Y: y, seqlen: s})[0])

            temp = [[Output[i], i + 1] for i in range(len(Output))]

            temp = sorted(temp, reverse=True)
            print(temp[0:10])
            #
            O = []
            for i in range(10):
                O.append(Dict[temp[i][1]])

            print(time.time() - tim)

            print(O)

            print(Dict[np.array(sess.run(out, feed_dict={X: x, Y: y, seqlen: s})).argmax() + 1])

            print(out)
            for i in range(len(out)):
                if out[i] == max(out):
                    print(i)

        except:
            continue


