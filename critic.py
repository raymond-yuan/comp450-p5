import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers

# tf.enable_eager_execution()


class CriticModel(Model):
    def __init__(self, env_dim, act_dim):
        super(CriticModel, self).__init__()
        self.d1 = layers.Dense(256, input_shape=(env_dim,), activation='relu')
        self.d2 = layers.Dense(256, activation='relu')

        self.a1 = layers.Dense(256, input_shape=(4,))

        self.d3 = layers.Dense(300, activation='relu')
        self.out_layer = layers.Dense(1, kernel_initializer=RandomUniform())

    def call(self, inputs):
        inputs, action = inputs
        # print(inputs.shape)

        x = self.d1(inputs)
        x = self.d2(x)

        a = self.a1(action)
        # print("ACTION", a.shape)
        #
        # print("feats", x.shape)
        c = tf.concat([x, a], axis=1)
        x = self.d3(c)
        out = self.out_layer(x)
        return out


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = CriticModel(self.env_dim, self.act_dim)
        self.target_model = CriticModel(self.env_dim, self.act_dim)
        print(self.env_dim)
        dummy_init_state = tf.constant(np.ones(self.env_dim), dtype=tf.float64)
        dummy_init_acts = tf.constant(np.ones((1,) + (self.act_dim,)), dtype=tf.float64)

        self.opt = tf.train.AdamOptimizer(self.lr)
        # self.model.compile(tf.train.AdamOptimizer(self.lr), 'mse')
        # self.target_model.compile(tf.train.AdamOptimizer(self.lr), 'mse')
        self.model([dummy_init_state, dummy_init_acts])
        self.target_model([dummy_init_state, dummy_init_acts])

        # Function to compute Q-value gradients (Actor Optimization)
        # self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        with tf.GradientTape() as t:
            out = self.model([states, actions])
        return t.gradient(out, actions)

    def target_predict(self, inputs):
        """ Predict Q-Values using the target network
        """
        return self.target_model(inputs)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        with tf.GradientTape() as t:
            out = self.model([states, actions])
            loss = tf.reduce_mean(tf.square(out - critic_target))
        grads = t.gradient(loss, self.model.weights)
        self.opt.apply_gradients(zip(grads, self.model.weights))

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)