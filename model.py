# !/usr/bin/python3
# _*_coding: utf-8_*_

import time
import tensorflow as tf

flags = tf.flags

flags.DEFINE_bool("use_fp16", False, "Whether use the type of float16")

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

def Config(object):
    hidden_size = 200
    keep_prob = 0.5
    num_layers = 2
    bath_size = 20
    vocab_size = 8000
    num_steps = 20
    max_grad_norm = 5
class PTB(object):
    def init(self, config, is_training):
        self.size = config.hidden_size
        self.keep_prob = config.keep_prob
        self.is_training = is_training
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.num_steps = config.num_setps
        self._input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self._targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.max_grad_norm = config.max_grad_norm

        cell = _get_lstm_model()
        self._initial_state = cell.zero_state(self.batch_size, data_type())

        inputs = self._get_inputs()
        outputs = self._output()
        self._loss, self._final_state = self._cost(outputs)
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), self.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grad, tvars))

    def run_epoch(session, model, data, eval_op, verbose=False):

        epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
        start_time = time.time
        costs = 0.0
        state = session.run(model.initial_state)


    def _get_lstm_model(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=0.0, state_is_tuple=True)

        if (config.keep_prob <  and self.is_traing):
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)
        return cell

    def _get_inputs(self):
        embedding = tf.get_variable("embedding", [self.vovab_size, self.size], dtype=data_type())
        inputs = tf.nn.embedding_lookup(embedding, tf._input_data)

        if(self.is_training and self.keep_prob < 1):
            inputs = tf.nn.dropout(inputs, self.keep_prob)

    def _output(self):
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if (time_step > 0):
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        outputs = tf.reshape(tf.concat(1, outputs), [-1, self.size])
        return outputs

    def _cost(self, outputs):
        softmax_w = tf.get_variable("softmax_w", [self.size, self.vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=data_type())
        logits = tf.matmul(outputs, softmax_w) + softmax_b

        cost = tf.contrib.seq2seq.sequence_loss(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([self.batch_size * self.num_steps], dtype=data_type())]
        )
        cost = tf.reduce_sum(cost) / self.batch_size
        return cost, state
