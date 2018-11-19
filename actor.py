import numpy as np
import tensorflow as tf

from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
# from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten

# tf.enable_eager_execution()

class ActorModel(Model):
    def __init__(self, env_dim, act_range):
        super(ActorModel, self).__init__()
        self.d1 = layers.Dense(400, input_shape=(env_dim,), activation='relu')
        self.g1 = layers.GaussianNoise(1.0)
        # self.f = layers.Flatten()
        self.d2 = layers.Dense(300, activation='relu')
        self.g2 = layers.GaussianNoise(1.0)        #
        self.out_layer = layers.Dense(4, activation='tanh', kernel_initializer=RandomUniform())
        self.act_range = act_range

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.g1(x)
        # x = self.f(x)
        x = self.d2(x)
        x = self.g2(x)
        x = self.out_layer(x)
        x = tf.multiply(x, self.act_range)
        return x


class Actor:
    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.model = ActorModel(self.env_dim, self.act_range)
        self.target_model = ActorModel(self.env_dim, self.act_range)
        #  initialize model
        dummy_init = tf.constant(np.ones((1,) + self.env_dim), dtype=tf.float64)
        self.model(dummy_init)
        self.target_model(dummy_init)

        self.opt = tf.train.AdamOptimizer(self.lr)

    def predict(self, state):
        """ Action prediction
        """
        return self.model(tf.constant(np.expand_dims(state, axis=0)))

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model(tf.constant(inp))

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, grads):
        with tf.GradientTape() as t:
            out = self.model(states)
        actor_grads = t.gradient(out, self.model.trainable_weights, -grads)
        self.opt.apply_gradients(zip(actor_grads, self.model.trainable_weights))
