import gym
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tqdm import tqdm, trange
import os
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

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Render flag')
parser.add_argument('--raw_feats', action='store_true', help='Render flag')
parser.add_argument('--disp', action='store_true', help='Render flag')
parser.add_argument('--gridsearch', action='store_true', help='Render flag')
parser.add_argument('--eval_random', action='store_true', help='Render flag')
parser.add_argument('--model', type=str, default='')
parser.add_argument('--eval_model', type=str, default='')
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()


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
                 raw=True,
                 concat=False):
        super(ActorCriticFetch, self).__init__()
        self.core1 = layers.Dense(128, activation='relu')
        self.core2 = layers.Dense(256, activation='relu')

        self.dist_thresh = 0.05
        self.raw_feats, self.concat_feats = raw, concat
        self.pre_mu = layers.Dense(512, activation='relu')
        self.mu_layer = layers.Dense(1, activation='tanh')

        self.pre_sigma = layers.Dense(512, activation='relu')
        self.sigma_layer = layers.Dense(1)

        # self.pre_value1 = layers.Dense(256, activation='relu')
        self.pre_value2 = layers.Dense(512, activation='relu')
        self.value_layer = layers.Dense(1, name='value')
        self.call(env.reset()['observation'])

        self.save_weights('init_weights.h5')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    def call(self, inputs):
        x = featurize_state(inputs, self.raw_feats, self.concat_feats)
        x = tf.expand_dims(x, 0)

        x = self.core1(x)
        x = self.core2(x)

        # px = self.pre_value1(x)
        px = self.pre_value2(x)
        self.value = self.value_layer(px)
        self.value = tf.squeeze(self.value)

        pm = self.pre_mu(x)
        self.mu = self.mu_layer(pm)

        ps = self.pre_sigma(x)
        ps = self.sigma_layer(ps)
        # self.sigma = tf.nn.softplus(self.sigma_layer(x)) + 1e-6
        self.sigma = tf.nn.softplus(ps) + 1e-10

        self.mu = tf.squeeze(self.mu)
        self.sigma = tf.squeeze(self.sigma)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        action = self.normal_dist.sample(1)
        self.action = tf.squeeze(action)
        self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
        return self.action, self.value

    def update(self, tape):
        # grads = tape.gradient(self.a_loss + 0.5 * self.v_loss, self.p_weights + self.v_weights)
        # self.p_opt.apply_gradients(zip(grads, self.p_weights + self.v_weights))
        # policy_grads = tape.gradient(self.a_loss, self.p_weights)
        # policy_grads = [-g for g in policy_grads]
        # value_grads = tape.gradient(self.v_loss, self.v_weights)
        # value_grads = [-g for g in value_grads]
        grads = tape.gradient(self.loss, self.weights)

        # flat_grads = []
        # for g in grads:
        #     flat_grads.extend(tf.reshape(g, [-1]))

        # print('p max grads:' , tf.reduce_max(flat_grads))
        # print('p min flat_grads:' ,tf.reduce_max(flat_grads))
        # print('p mean flat_grads:' ,tf.reduce_mean(flat_grads))

        self.p_opt.apply_gradients(zip(grads, self.weights))
        # self.p_opt.apply_gradients(zip(policy_grads, self.p_weights))
        # self.v_opt.apply_gradients(zip(value_grads, self.v_weights))

    def get_policy_loss(self, policy_target):
        self.a_loss = self.normal_dist.log_prob(self.action) * tf.stop_gradient(policy_target)
        # Add cross entropy cost to encourage exploration
        self.a_loss -= 1e-1 * self.normal_dist.entropy()
        return self.a_loss

    def get_value_loss(self, value_target):
        self.v_loss = tf.reduce_mean(tf.square(self.value - value_target))  # -tf.math.squared_difference(self.value, value_target)
        return self.v_loss


class MultiAC():
    def __init__(self, env):
        self.env = env
        self.a1 = ActorCriticFetch(args.raw_feats)
        self.a2 = ActorCriticFetch(args.raw_feats)
        self.a3 = ActorCriticFetch(args.raw_feats)
        self.a4 = ActorCriticFetch(args.raw_feats)
        self.models = [self.a1, self.a2, self.a3, self.a4]
        self.dist_thresh = 0.1

    def call(self, inputs):
        actions = []
        values = []
        for m in self.models:
            a, v = m(inputs)
            actions.append(a)
            values.append(v)
        return actions, values

    def train(self, num_eps=100,
              render=False,
              model_start='',
              model_save='fetchReachMulti.h5',
              custom_r=False,
              v_lr=1.0,
              p_lr=1.0,
              verbose=1):
        best_avg_model_name = 'best_avg-' + model_save

        for i, m in enumerate(self.models):
            # m.p_opt = tf.train.AdamOptimizer(learning_rate=p_lr)
            m.p_opt = tf.train.RMSPropOptimizer(learning_rate=p_lr, epsilon=0.1)
            # self.p_opt = tf.train.GradientDescentOptimizer(learning_rate=p_lr)
            # m.v_opt = tf.train.AdamOptimizer(learning_rate=v_lr)

        self.num_eps = num_eps
        best_r = -float('inf')
        best_avg = -float('inf')

        num_steps = 200
        avg_r_ep = 0
        prev_actions = np.zeros(4, dtype=np.float64)

        if verbose == 0:
            ep_iter = tqdm(range(self.num_eps))
        else:
            ep_iter = range(self.num_eps)

        solve_count = 0
        avg = 0
        for ep in ep_iter:
            tr = range(num_steps)
            if verbose == 1:
                tr = tqdm(tr)

            state = self.env.reset()

            total_r = 0
            avg_value = 0
            actions = np.zeros(4, dtype=np.float64)

            for t in tr:
                with tf.GradientTape(persistent=True) as tape:
                    action, value = self.call(state['observation'])

                    actions += action
                    if render:
                        self.env.render()
                    next_state, r, done, info = self.env.step(action)
                    d = fetch_env.goal_distance(next_state['achieved_goal'], next_state['desired_goal'])
                    done = d <= self.dist_thresh and t > 1
                    if custom_r:
                        r = (self.dist_thresh / (d + 1e-6))
                        r = min(r, 1.0)
                        # substitute_goal = state['achieved_goal'].copy()
                        # r = env.compute_reward(state['achieved_goal'], substitute_goal, info)

                    if done:
                        solve_count += 1
                        print('solved')
                        if custom_r:
                            r += 5
                    total_r += r
                    avg_value += tf.reduce_mean(value)
                    self.loss = []
                    for i, m in enumerate(self.models):
                        td_target = r + 0.99 * m(next_state['observation'])[1]
                        td_error = td_target - value[i]
                        vloss = m.get_value_loss(td_target)
                        ploss = tf.reduce_mean(m.get_policy_loss(td_error))
                        m.loss = vloss + ploss
                        self.loss.append(m.loss)
                        # if not custom_r:
                        #     m.loss *= -1

                for m in self.models:
                    m.update(tape)
                self.loss = tf.reduce_mean(self.loss)

                if verbose == 1:
                    tr.set_description("Ep {}/{} | "
                                       "Loss {:.3f} | "
                                       "Total Reward {:.3f} | "
                                       "Avg Value {:.3f} | "
                                       "Solve Ratio {:.3f} | "
                                       "Avg Reward/Epoch {:.3f}".format(ep + 1, self.num_eps,
                                                                        self.loss,
                                                                        total_r,
                                                                        avg_value / (t + 1),
                                                                        solve_count / (ep + 1),
                                                                        avg_r_ep
                                                                        ))
                if done:
                    break

                state = next_state

            if verbose == 0:
                ep_iter.set_description("Solve percentage: {:.3f}".format(solve_count / ep))

            if done and total_r < avg:
                total_r = avg + 5

            if avg_r_ep == 0:
                avg_r_ep = total_r
            else:
                avg_r_ep = avg_r_ep * 0.99 + total_r * 0.01
            if fetch_env.goal_distance(actions, prev_actions) < 0.1:
                print('Possible error: actions same as previous state {}\n'.format(actions / num_steps))
            prev_actions = actions
            avg = avg_r_ep
            if avg >= best_avg and ep > 10:
                print(f'\nSaving best average model with reward of {avg} to {best_avg_model_name}')
                best_avg = avg
                for i, m in enumerate(self.models):
                    m.save_weights('pretrained/' + str(i) + best_avg_model_name)

            if total_r >= best_r:
                print(f'\nSaving best model with reward of {total_r} to {model_save}')
                best_r = total_r
                for i, m in enumerate(self.models):
                    m.save_weights('pretrained/' + str(i) + model_save)
        for i, m in enumerate(self.models):
            m.save_weights('pretrained/last_' + str(i) + model_save)

    def eval(self, model_name='', random=False):
        if not random:
            for i, m in self.models:
                m.load_weights('pretrained/' + str(i) + model_name)
        score = 0
        solve_count = 0
        tr = tqdm(range(200))
        for ep in tr:
            state = self.env.reset()
            tr.set_description("Solve percentage: {:.3f}".format(solve_count / (ep + 1)))
            for t in range(200):
                if random:
                    a = self.env.action_space.sample()
                else:
                    a, v = self.call(state['observation'])

                state, r, done, info = self.env.step(a)
                d = fetch_env.goal_distance(state['achieved_goal'], state['desired_goal'])
                done = d <= self.dist_thresh
                if done:
                    solve_count += 1
                    break
                score += r

        return score / 100.0

    def disp_final(self):
        self.load_weights('pretrained/best_avg-negLoss-fetchReach-plr:0.01-vlr:0.01-raw:True-customr:True.h5')
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

if __name__ == '__main__':
    if not os.path.exists('pretrained'):
        os.makedirs('pretrained/')

    a = MultiAC(env)
    if args.disp:

        a.disp_final()
    elif args.eval_model != '':
        a = ActorCriticFetch(env, raw=True)
        print(a.eval(model_name=args.eval_model))
    elif args.eval_random:
        print(a.eval(random=True))

    else:
        a.train(p_lr=1e-4,
                num_eps=args.epochs,
                render=args.render,
                model_start=args.model,
                model_save='fetchReachMulti2.h5',
                custom_r=True)
