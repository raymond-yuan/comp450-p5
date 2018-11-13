import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tqdm import tqdm, trange
from gym.envs.robotics.fetch.push import FetchPushEnv
import argparse
import collections

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

tf.enable_eager_execution()
env = gym.make('FetchPush-v1')
# env.reward_type = 'dense'
# env.target_offset=1.0
# _max_episode_steps = 100
env = FetchPushEnv(reward_type='dense')
env._max_episode_steps = 100
observation_examples = np.array([env.reset()['observation'] for _ in range(10000)])
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


class ActorCriticFetch(models.Model):
    def __init__(self,
                 env):
        super(ActorCriticFetch, self).__init__()
        self.env = env

        self.pre_mu = layers.Dense(128, activation='relu')
        self.mu_layer = layers.Dense(4)
        self.pre_sigma = layers.Dense(128, activation='relu')
        self.sigma_layer = layers.Dense(4)

        self.pre_value = layers.Dense(64, activation='relu')
        self.value_layer = layers.Dense(1, name='value')
        self.call(env.reset()['observation'])

        self.p_weights = self.pre_mu.weights + self.mu_layer.weights + self.pre_sigma.weights + self.sigma_layer.weights
        # self.p_weights = self.mu_layer.weights + self.sigma_layer.weights
        self.v_weights = self.pre_value.weights + self.value_layer.weights

        self.p_opt = tf.train.AdamOptimizer(learning_rate=0.01)
        self.v_opt = tf.train.AdamOptimizer(learning_rate=0.1)

    def call(self, inputs):
        # x = featurize_state(inputs)
        x = tf.expand_dims(inputs, 0)

        px = self.pre_value(x)
        self.value = self.value_layer(px)
        self.value = tf.squeeze(self.value)

        pm = self.pre_mu(x)
        self.mu = self.mu_layer(pm)

        ps = self.sigma_layer(self.pre_sigma(x))
        # self.sigma = tf.nn.softplus(self.sigma_layer(x)) + 1e-6
        self.sigma = tf.nn.softplus(ps) + 1e-6

        self.mu = tf.squeeze(self.mu)
        self.sigma = tf.squeeze(self.sigma)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        self.action = tf.squeeze(self.normal_dist.sample([1]))
        self.action = tf.clip_by_value(self.action, env.action_space.low, env.action_space.high)
        return self.action, self.value

    def update(self, tape):
        policy_grads = tape.gradient(self.a_loss, self.p_weights)
        value_grads = tape.gradient(self.v_loss, self.v_weights)
        self.p_opt.apply_gradients(zip(policy_grads, self.p_weights))
        self.v_opt.apply_gradients(zip(value_grads, self.v_weights))

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

    def train(self, num_eps=100, render=False):
        # self.load_weights('mtnCar.h5')
        self.num_eps = num_eps

        best_r = -float('inf')

        num_steps = 100

        for ep in range(self.num_eps):
            tr = tqdm(range(num_steps))
            # tr = tqdm(itertools.count())
            state = self.env.reset()
            total_r = 0
            avg_value = 0
            for t in tr:
                with tf.GradientTape(persistent=True) as tape:
                    a, v = self.call(state['observation'])

                    if render:
                        self.env.render()
                    next_state, r, done, info = self.env.step(a)
                    total_r += r
                    avg_value += v.numpy()
                    td_target = r + 0.99 * self.call(next_state['observation'])[1]
                    td_error = td_target - v
                    self.get_value_loss(td_target)
                    self.get_policy_loss(td_error)
                self.update(tape)

                # grads = self.get_grads(tape, td_error, td_target)
                # self.optimizer.apply_gradients(zip(grads, self.weights))
                tr.set_description("Ep {}/{} | "
                                   "Total Reward {:.3f} | "
                                   "Avg Reward {:.3f} | "
                                   "Avg Value {:.3f}"
                                   "step {}".format(ep, self.num_eps, total_r, total_r / (t + 1), avg_value / (t + 1), t))
                if done: break

                state = next_state
            if total_r >= best_r:
                print(f'\nSaving best model with reward of {total_r}')
                best_r = total_r
                self.best_weights = self.weights
                self.save_weights('fetchReach.h5')

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Render flag')
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

if __name__ == '__main__':
    a = ActorCriticFetch(env)
    a.train(num_eps=args.epochs,
            render=args.render)
# def policy(observation, desired_goal):
#     # Here you would implement your smarter policy. In this case,
#     # we just sample random actions.
#     return env.action_space.sample()
#
# while not done:
#     env.render()
#     action = policy(obs['observation'], obs['desired_goal'])
#     obs, reward, done, info = env.step(action)
#
#     # If we want, we can substitute a goal here and re-compute
#     # the reward. For instance, we can just pretend that the desired
#     # goal was what we achieved all along.
#     substitute_goal = obs['achieved_goal'].copy()
#     substitute_reward = env.compute_reward(
#         obs['achieved_goal'], substitute_goal, info)
#     print('reward is {}, substitute_reward is {}'.format(
#         reward, substitute_reward))
