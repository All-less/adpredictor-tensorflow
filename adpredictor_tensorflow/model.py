# coding: utf-8
import tensorflow as tf

from . import bayesian
from . import utils


class adPredictor(object):

    def __init__(self, config, session):
        self._config = config
        self._sess = session
        self._dists = tf.Variable(utils.get_dists_init(config), name='dists')
        self._build_op()

    def _build_op(self):
        self._X = tf.placeholder(tf.int32, (self._config.num_features,), name='X')
        self._y = tf.placeholder(tf.int32, name='y')
        new_dists = tf.py_func(self._py_fit, [ self._dists, self._X, self._y ], tf.float32, name='train')
        self._train_op = tf.assign(self._dists, new_dists)
        self._pred_op = tf.py_func(self._py_predict, [ self._dists, self._X ], tf.float64, name='pred')

    def fit(self, features, labels):
        self._sess.run(tf.global_variables_initializer())
        for X, y in zip(features, labels):
            self._sess.run(self._train_op, feed_dict={ self._X: X, self._y: y })

    def predict(self, X):
        return self._sess.run(self._pred_op, feed_dict={ self._X: X })

    def _py_fit(self, dists, X, y):
        return bayesian.update(dists, X, y, self._config.beta, self._config.epsilon)

    def _py_predict(self, dists, X):
        return bayesian.predict(dists, X, self._config.beta)

