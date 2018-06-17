import tensorflow as tf
import numpy as np

# Helper to make the output consistent
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def build_graph(MC: dict, AC: dict):
    reset_graph()
    g = tf.Graph()
    with g.as_default():

        #### model architecture
        with tf.name_scope("inputs"):
            # TODO: input layer logic
            pass
        with tf.name_scope("hidden"):
            # TODO: hidden layer logic
            pass
        with tf.name_scope("logits"):
            # TODO: output layer logic
            pass

        #### loss logic
        with tf.name_scope("loss"):
            # TODO: loss logic
            pass

        #### optimizer
        with tf.name_scope("train"):
            # TODO: optimizer
            # TODO: training operation
            pass

        #### saver
        with tf.name_scope("save_session"):
            # TODO: implement
            # init_global = tf.global_variables_initializer()
            # init_local = tf.local_variables_initializer()
            # saver = tf.train.Saver()
            pass

        #### metrics
        with tf.name_scope("metrics"):
            pass

    return g
