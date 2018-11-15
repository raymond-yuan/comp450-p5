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
from gym.envs.robotics.fetch.reach import FetchReachEnv
from gym.envs.robotics import fetch_env
import argparse
import collections

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

tf.enable_eager_execution()
# env = gym.make('FetchReach-v1')
# env.reward_type = 'dense'
# env.target_offset=1.0
# _max_episode_steps = 100
env = FetchReachEnv(reward_type='dense')
env.max_episode_steps = 100
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


def featurize_state(state, raw=False, concat=False):
    """
    Returns the featurized representation for a state.
    """
    if raw:
        return state
    # state[np.isnan(state)] = 0
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    if concat:
        out = np.concatenate((featurized[0], state))
    else:
        out = featurized[0]
    return out


class ActorCriticFetch(models.Model):
    def __init__(self,
                 env,
                 raw=True,
                 concat=False):
        super(ActorCriticFetch, self).__init__()
        self.env = env
        self.raw_feats, self.concat_feats = raw, concat
        self.pre_mu = layers.Dense(128, activation='relu', kernel_initializer='zeros')
        self.mu_layer = layers.Dense(3, kernel_initializer='zeros')
        self.pre_sigma = layers.Dense(128, activation='relu', kernel_initializer='zeros')
        self.sigma_layer = layers.Dense(3, kernel_initializer='zeros')

        self.pre_value = layers.Dense(128, activation='relu', kernel_initializer='zeros')
        self.value_layer = layers.Dense(1, name='value', kernel_initializer='zeros')
        self.call(env.reset()['observation'])

        self.p_weights = self.pre_mu.weights + self.mu_layer.weights + self.pre_sigma.weights + self.sigma_layer.weights
        # self.p_weights = self.mu_layer.weights + self.sigma_layer.weights
        self.v_weights = self.pre_value.weights + self.value_layer.weights

        self.save_weights('init_weights.h5')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    def call(self, inputs):
        x = featurize_state(inputs, self.raw_feats, self.concat_feats)
        x = tf.expand_dims(x, 0)

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
        self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
        return self.action, self.value

    def update(self, tape):
        # grads = tape.gradient(self.a_loss + 0.5 * self.v_loss, self.p_weights + self.v_weights)
        # self.p_opt.apply_gradients(zip(grads, self.p_weights + self.v_weights))
        policy_grads = tape.gradient(self.a_loss, self.p_weights)
        # policy_grads = [-g for g in policy_grads]
        value_grads = tape.gradient(self.v_loss, self.v_weights)
        # value_grads = [-g for g in value_grads]
        self.p_opt.apply_gradients(zip(policy_grads, self.p_weights))
        self.v_opt.apply_gradients(zip(value_grads, self.v_weights))

    def get_policy_loss(self, policy_target):
        self.a_loss = self.normal_dist.log_prob(self.action) * policy_target
        # Add cross entropy cost to encourage exploration
        self.a_loss -= 1e-1 * self.normal_dist.entropy()
        return self.a_loss

    def get_value_loss(self, value_target):
        self.v_loss = tf.math.squared_difference(self.value, value_target)
        return self.v_loss

    def get_grads(self, t, policy_target, value_target):
        self.a_loss = self.normal_dist.log_prob(self.action) * policy_target
        # Add cross entropy cost to encourage exploration
        self.a_loss -= 1e-1 * self.normal_dist.entropy()

        self.v_loss = tf.math.squared_difference(self.value, value_target)

        self.loss = self.a_loss + 0.5 * self.v_loss
        return t.gradient(self.loss, self.weights)

    def train(self, num_eps=100,
              render=False,
              model_start='',
              model_save='fetchReach.h5',
              custom_r=False,
              v_lr=1.0,
              p_lr=1.0):

        self.p_opt = tf.train.AdamOptimizer(learning_rate=p_lr)
        self.v_opt = tf.train.AdamOptimizer(learning_rate=v_lr)
        # self.load_weights('mtnCar.h5')
        if model_start:
            self.load_weights(model_start)

        self.num_eps = num_eps

        best_r = -float('inf')
        best_avg = -float('inf')

        num_steps = 100
        avg_r_ep = 0
        for ep in range(self.num_eps):
            tr = tqdm(range(num_steps))
            # tr = tqdm(itertools.count())
            state = self.env.reset()
            self.env.distance_threshold = 5
            total_r = 0
            avg_value = 0
            for t in tr:
                with tf.GradientTape(persistent=True) as tape:
                    a, v = self.call(state['observation'])
                    a = [0] + list(a.numpy())

                    if render:
                        self.env.render()
                    next_state, r, done, info = self.env.step(a)

                    if custom_r:
                        r += (0.1 / (fetch_env.goal_distance(next_state['achieved_goal'], next_state['desired_goal']) + 1e-5))
                    if done:
                        print('solved')
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
                                   "Frame Reward {:.3f} | "
                                   "Total Reward {:.3f} | "
                                   "Avg Value {:.3f} | "
                                   "Avg Reward/Epoch {:.3f}".format(ep + 1, self.num_eps,
                                                                    r,
                                                                    total_r,
                                                                    avg_value / (t + 1),
                                                                    avg_r_ep / (ep + 1)))
                                   # "Avg Reward {:.3f} | "
                if done: break

                state = next_state
            avg_r_ep += total_r

            if avg_r_ep / (ep + 1) >= best_avg:
                best_avg = avg_r_ep / (ep + 1)
                self.save_weights('best_avg-' + model_save)

            if total_r >= best_r:
                print(f'\nSaving best model with reward of {total_r}')
                best_r = total_r
                self.best_weights = self.weights
                self.save_weights(model_save)

    def eval(self, model_name):
        self.load_weights(model_name)
        score = 0
        for ep in range(100):
            state = self.env.reset()
            for t in range(100):
                a, v = self.call(state['observation'])
                a = [0] + list(a.numpy())

                state, r, done, info = self.env.step(a)
                if done: break
                score += r
        return score / 100.0

    def disp_final(self):
        self.load_weights('fetchReach-lr:0.3-raw:False-customr:True.h5')
        state = self.env.reset()
        tr = tqdm(itertools.count())
        total_r = 0
        for t in tr:
            a, v = self.call(state['observation'])
            a = [0] + list(a.numpy())
            self.env.render()
            state, r, done, info = self.env.step(a)
            if done: break
            total_r += r
            tr.set_description(f"step {t} || Reward {total_r}")

            if done: break
        self.env.close()

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Render flag')
parser.add_argument('--disp', action='store_true', help='Render flag')
parser.add_argument('--gridsearch', action='store_true', help='Render flag')
parser.add_argument('--model', type=str, default='')
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

if __name__ == '__main__':
    a = ActorCriticFetch(env, raw=False)
    if args.disp:
        a.disp_final()
    elif args.gridsearch:
        best_score = -float('inf')
        for lr in [0.01, 0.1, 0.3, 0.5, 1.0]:
            for r in [True, False]:
                for custom_r in [True, False]:
                    a = ActorCriticFetch(env, raw=r, concat=False)
                    model_save = f'fetchReach-lr:{lr}-raw:{r}-customr:{custom_r}.h5'
                    print(model_save)
                    a.load_weights('init_weights.h5')
                    a.train(p_lr=lr,
                            v_lr=lr,
                            num_eps=700,
                            custom_r=custom_r,
                            model_save=model_save
                            )
                    score = a.eval(model_save)
                    if score > best_score:
                        print('Using best model')
                        best_score = score
                        best_model = model_save
                        print(f'Best model: {model_save} with score {score}')

                    score = a.eval('best_avg-' + model_save)
                    if score > best_score:
                        print('Using best average model')
                        best_score = score
                        best_model = 'best_avg-' + model_save
                        print(f"Best model: {'best_avg-' + model_save} with score {score}")
        print()
        print(best_model)
        print(best_score)

    else:
        a.train(
                num_eps=args.epochs,
                render=args.render,
                model_start=args.model)



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
