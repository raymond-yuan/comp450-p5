import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tqdm import tqdm, trange
import collections

import sklearn.pipeline
import sklearn.preprocessing

from sklearn.kernel_approximation import RBFSampler
tf.enable_eager_execution()
env = gym.envs.make("MountainCarContinuous-v0")

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    # state[np.isnan(state)] = 0
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]

class ActorCritic(models.Model):
    def __init__(self, 
                 env,
                 num_eps=50):
        super(ActorCritic, self).__init__()
        self.env = env
        self.num_eps = num_eps
        self.mu_layer = layers.Dense(1, kernel_initializer='zeros')
        self.sigma_layer = layers.Dense(1, kernel_initializer='zeros')
        self.value_layer = layers.Dense(1, name='value', kernel_initializer='zeros')

        # init model
        self.call(env.observation_space.sample())
        self.v_optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        self.p_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

        self.actor_ws = self.mu_layer.weights + self.sigma_layer.weights
        self.critic_ws = self.value_layer.weights
    
    def call(self, inputs):
        x = featurize_state(inputs)
        x = tf.expand_dims(x, 0)
        
        self.value = self.value_layer(x)
        self.value = tf.squeeze(self.value)
        
        self.mu = self.mu_layer(x)
        self.sigma = tf.nn.softplus(self.sigma_layer(x))

        self.mu = tf.squeeze(self.mu)
        self.sigma = tf.squeeze(self.sigma)
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.normal_dist._sample_n(1)
        self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
        return self.action, self.value

    def update(self, tape):
        policy_grads = tape.gradient(self.a_loss, self.actor_ws)
        value_grads = tape.gradient(self.v_loss, self.critic_ws)
        self.p_optimizer.apply_gradients(zip(policy_grads, self.actor_ws))
        self.v_optimizer.apply_gradients(zip(value_grads, self.critic_ws))

    def get_policy_loss(self, policy_target):
        self.a_loss = -self.normal_dist.log_prob(self.action) * policy_target
        # Add cross entropy cost to encourage exploration
        self.a_loss -= 1e-1 * self.normal_dist.entropy()
        return self.a_loss


    def get_value_loss(self, value_target):
        self.v_loss = tf.math.squared_difference(self.value, value_target)
        return self.v_loss

    def get_grads(self, t, policy_target, value_target):
        self.a_loss = -self.normal_dist.log_prob(self.action) * policy_target
        # Add cross entropy cost to encourage exploration
        self.a_loss -= 1e-1 * self.normal_dist.entropy()
        
        self.v_loss = tf.math.squared_difference(self.value, value_target)
        
        self.loss = 0.01 * self.a_loss + self.v_loss
        return t.gradient(self.loss, self.weights)
#         actor_grads = t.gradient(self.a_loss, self.actor_ws)
#         value_grads = t.gradient(self.v_loss, self.critic_ws)
#         return actor_grads, value_grads

    def disp_final(self):
        self.load_weights('mtnCar.h5')
        state = self.env.reset()
        tr = tqdm(itertools.count())
        total_r = 0
        for t in tr:
            a, v = self.call(state)
            self.env.render()
            next_state, r, done, info = self.env.step(a)
            total_r += r
            tr.set_description(f"step {t} || Reward {total_r}")

            if done: break
        self.env.close()

    def train(self):
        # self.load_weights('mtnCar.h5')
        best_r = -float('inf')
        for ep in range(self.num_eps):
            tr = tqdm(itertools.count())
            state = self.env.reset()
            total_r = 0
            for t in tr:
                with tf.GradientTape(persistent=True) as tape:
                    a, v = self.call(state)
                    self.env.render()
                    next_state, r, done, info = self.env.step(a)
                    total_r += r
                    td_target = r + 0.99 * self.call(next_state)[1]
                    td_error = td_target - v
                    self.get_value_loss(td_target)
                    self.get_policy_loss(td_error)
                self.update(tape)

                    # grads = self.get_grads(tape, td_error, td_target)
                # self.optimizer.apply_gradients(zip(grads, self.weights))
                tr.set_description(f"Ep {ep}/{self.num_eps} | Reward {total_r} | step {t}")
                if done: break
                

                state = next_state
            if total_r >= best_r:
                print(f'\nSaving best model with reward of {total_r}')
                best_r = total_r
                self.best_weights = self.weights
                self.save_weights('mtnCar.h5')

if __name__ == '__main__':
    a = ActorCritic(env)

    a.train()
    a.disp_final()
