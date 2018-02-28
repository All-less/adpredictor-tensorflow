# coding: utf-8
import tensorflow as tf

from model import adPredictor


def get_flags():
    flags = tf.app.flags
    flags.DEFINE_integer('num_features', 3,
                         'number of features')
    # We assume all features are discrete and within 1 ... <feature_max>
    flags.DEFINE_integer('feature_max', 2,
                         'maximal possible value of a feature')
    flags.DEFINE_float('beta', 0.05,
                       '')
    flags.DEFINE_float('epsilon', 0.05,
                       '')
    flags.DEFINE_float('prior_prob', 0.5,
                       '')
    return flags.FLAGS

def train(features, labels):
    """
    Sample input:
        features = [ [1, 2, 2], [2, 1, 1] ]
        labels = [ -1, 1 ]  # labels must be in { -1, 1 }
    """
    with tf.Session() as sess:
        model = adPredictor(get_flags(), sess)
        model.fit(features, labels)

if __name__ == '__main__':
    train([ [1, 2, 2], [2, 1, 1] ], [ -1, 1 ])
