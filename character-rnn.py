import tensorflow as tf
import numpy as np



SEQUENCE_LEN = 24
ELEMENT_LEN = 1

TRAINING_BATCH_SIZE = 128
N_TRAINING_EPOCHS = 10


def text_to_intarr(text):
    intarr = []
    for char in text:
        intarr.append([ord(char)])
    return intarr

def intarr_to_seqs(intarr, xlen, ylen):
    xseqs = []
    yseqs = []
    for i in range(0, len(intarr), xlen+ylen):
        xseqs.append(intarr[i:i+xlen])
        yseqs.append(intarr[i+xlen:i+xlen+ylen])

    # remove the final element in the sequence - this is always a funny
    # shape, and causes issues when we try to pass it in as a tensor
    return xseqs[:-1], yseqs[:-1]

def batchify(seq, batch_size):
    batches = []
    for i in range(0, len(seq), batch_size):
        batches.append(seq[i:i+batch_size])
    return batches


# read in our text data
with open("the-last-question.txt") as file:
    text = file.read()

# preprocess
intarr = text_to_intarr(text)
xseqs, yseqs = intarr_to_seqs(intarr, SEQUENCE_LEN, 1)
train_X, train_y = batchify(xseqs, TRAINING_BATCH_SIZE),\
                   batchify(yseqs, TRAINING_BATCH_SIZE)


# (1) network inputs
with tf.variable_scope("input"):
    X = tf.placeholder(tf.float32, shape=[None, SEQUENCE_LEN, ELEMENT_LEN])
    X_transpose = tf.transpose(X, [1,0,2])
    X_reshape = tf.reshape(X_transpose, shape=[-1, ELEMENT_LEN])
    X_split = tf.split(X_reshape, SEQUENCE_LEN, 0)

# (2) network architecture - a single LSTM cell with 100 hidden units
with tf.variable_scope("lstm"):
    cell = tf.nn.rnn_cell.LSTMCell(100)
    outputs, state = tf.nn.static_rnn(cell, X_split, dtype=tf.float32)

with tf.variable_scope("fully-connected"):
    W = tf.get_variable("weights", dtype=tf.float32, shape=[100, 1])
    b = tf.get_variable("biases", dtype=tf.float32, shape=[1])
    pre_softmax = tf.matmul(outputs[-1], W) + b
    y_ = tf.nn.softmax(pre_softmax)

# (3) loss function
with tf.variable_scope("loss"):
    y = tf.placeholder(tf.float32, shape=[None, 1, 1])
    loss = tf.reduce_mean( (y-pre_softmax)**2 )

# (4) optimizer
with tf.variable_scope("optimizer"):
    optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

# (5) training loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(N_TRAINING_EPOCHS):
        epoch_loss = 0.0

        for i in range(len(train_X)):
            batch_X = train_X[i]
            batch_y = train_y[i]

            batch_loss, _ = sess.run([loss, optimizer],
                feed_dict={
                    X: train_X[i],
                    y: train_y[i]
                })

            epoch_loss += batch_loss

        print(epoch_loss)
