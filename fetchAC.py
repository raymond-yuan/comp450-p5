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
        self.dist_thresh = 0.05
        self.env = env
        self.raw_feats, self.concat_feats = raw, concat
        self.pre_mu1 = layers.Dense(512, activation='relu')
        self.pre_mu2 = layers.Dense(512, activation='relu')
        self.mu_layer = layers.Dense(4)

        self.pre_sigma1 = layers.Dense(512, activation='relu')
        self.pre_sigma2 = layers.Dense(512, activation='relu')
        self.sigma_layer = layers.Dense(4)

        self.pre_value1 = layers.Dense(512, activation='relu')
        self.pre_value2 = layers.Dense(512, activation='relu')
        self.value_layer = layers.Dense(1, name='value')
        self.call(env.reset()['observation'])

        self.p_weights = self.pre_mu1.weights + self.pre_mu2.weights + self.mu_layer.weights + self.pre_sigma1.weights + self.pre_sigma2.weights + self.sigma_layer.weights
        # self.p_weights = self.mu_layer.weights + self.sigma_layer.weights
        self.v_weights = self.pre_value1.weights + self.pre_value2.weights + self.value_layer.weights

        self.save_weights('init_weights.h5')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    def call(self, inputs):
        x = featurize_state(inputs, self.raw_feats, self.concat_feats)
        x = tf.expand_dims(x, 0)

        px = self.pre_value1(x)
        px = self.pre_value2(px)
        self.value = self.value_layer(px)
        self.value = tf.squeeze(self.value)

        pm = self.pre_mu1(x)
        pm = self.pre_mu2(pm)
        self.mu = self.mu_layer(pm)

        ps = self.pre_sigma1(x)
        ps = self.pre_sigma2(ps)
        ps = self.sigma_layer(ps)
        # self.sigma = tf.nn.softplus(self.sigma_layer(x)) + 1e-6
        self.sigma = tf.nn.softplus(ps) + 1e-6

        self.mu = tf.squeeze(self.mu)
        self.sigma = tf.squeeze(self.sigma)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        action = self.normal_dist.sample([1])
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
        self.a_loss = -self.normal_dist.log_prob(self.action) * tf.stop_gradient(policy_target)
        # Add cross entropy cost to encourage exploration
        self.a_loss -= 1e-1 * self.normal_dist.entropy()
        return self.a_loss

    def get_value_loss(self, value_target):
        self.v_loss = tf.math.squared_difference(self.value, value_target)
        return self.v_loss

    # def get_grads(self, t, policy_target, value_target):
    #     self.a_loss = self.normal_dist.log_prob(self.action) * policy_target
    #     # Add cross entropy cost to encourage exploration
    #     self.a_loss -= 1e-1 * self.normal_dist.entropy()
    #
    #     self.v_loss = tf.math.squared_difference(self.value, value_target)
    #
    #     self.loss = self.a_loss + 0.5 * self.v_loss
    #     return t.gradient(self.loss, self.weights)

    def train(self, num_eps=100,
              render=False,
              model_start='',
              model_save='fetchReach.h5',
              custom_r=False,
              v_lr=1.0,
              p_lr=1.0,
              verbose=1):
        best_avg_model_name = 'best_avg-' + model_save
        self.p_opt = tf.train.AdamOptimizer(learning_rate=p_lr)
        # self.p_opt = tf.train.GradientDescentOptimizer(learning_rate=p_lr)
        # self.v_opt = tf.train.AdamOptimizer(learning_rate=v_lr)
        # self.load_weights('mtnCar.h5')
        if model_start:
            self.load_weights(model_start)

        self.num_eps = num_eps
        best_r = -float('inf')
        best_avg = -float('inf')

        num_steps = 100
        avg_r_ep = 0
        prev_actions = np.zeros(4, dtype=np.float64)

        if verbose == 0:
            ep_iter = tqdm(range(self.num_eps))
        else:
            ep_iter = range(self.num_eps)

        solve_count = 0
        for ep in ep_iter:
            tr = range(num_steps)
            if verbose == 1:
                tr = tqdm(tr)

            # tr = tqdm(itertools.count())
            state = self.env.reset()
            # self.env.distance_threshold = 5
            total_r = 0
            avg_value = 0
            actions = np.zeros(4, dtype=np.float64)
            for t in tr:
                with tf.GradientTape(persistent=True) as tape:
                    a, v = self.call(state['observation'])

                    # a = [0] + list(a.numpy())
                    actions += a
                    if render:
                        self.env.render()
                    next_state, r, done, info = self.env.step(a)
                    d = fetch_env.goal_distance(next_state['achieved_goal'], next_state['desired_goal'])
                    done = d <= self.dist_thresh
                    if custom_r:
                        r = (self.dist_thresh / (d + 1e-6))
                        r = min(r, 1.0)

                    if done:
                        solve_count += 1
                        print('solved')
                    total_r += r
                    avg_value += v.numpy()
                    td_target = r + 0.99 * self.call(next_state['observation'])[1]
                    td_error = td_target - v
                    vloss = self.get_value_loss(td_target)
                    ploss = self.get_policy_loss(td_error)
                    self.loss = tf.reduce_mean(0.5 * vloss + ploss)
                self.update(tape)

                # grads = self.get_grads(tape, td_error, td_target)
                # self.optimizer.apply_gradients(zip(grads, self.weights))
                # "Frame Reward {:.3f} | "
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
                                                                        avg_r_ep / (ep + 1)
                                                                        ))
                                   # "Avg Reward {:.3f} | "
                if done:
                    break

                state = next_state

            if verbose == 0:
                ep_iter.set_description("Solve percentage: {:.3f}".format(solve_count / ep))

            avg_r_ep += total_r
            if fetch_env.goal_distance(actions, prev_actions) < 0.1:
                print('Possible error: actions same as previous state {}\n'.format(actions / num_steps))
            prev_actions = actions
            avg = avg_r_ep / (ep + 1)
            if avg >= best_avg and ep > 10:
                print(f'\nSaving best average model with reward of {avg} to {best_avg_model_name}')
                best_avg = avg
                self.save_weights('pretrained/' + best_avg_model_name)

            if total_r >= best_r:
                print(f'\nSaving best model with reward of {total_r} to {model_save}')
                best_r = total_r
                self.best_weights = self.weights
                self.save_weights('pretrained/' + model_save)
        self.save_weights('pretrained/last_' + model_save)

    def eval(self, model_name='', random=False):
        if not random:
            self.load_weights('pretrained/' + model_name)
        score = 0
        solve_count = 0
        tr = tqdm(range(100))
        for ep in tr:
            state = self.env.reset()
            tr.set_description("Solve percentage: {:.3f}".format(solve_count / (ep + 1)))
            for t in range(50):
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

if __name__ == '__main__':
    if not os.path.exists('pretrained'):
        os.makedirs('pretrained/')

    a = ActorCriticFetch(env, raw=args.raw_feats)
    if args.disp:

        a.disp_final()
    elif args.eval_model != '':
        a = ActorCriticFetch(env, raw=True)
        print(a.eval(model_name=args.eval_model))
    elif args.eval_random:
        print(a.eval(random=True))
    elif args.gridsearch:
        best_score = -float('inf')
        for plr in [0.01, 0.1, 1.0, 5]:
            for vlr in [0.1, 1.0, 5]:
                # for r in [True, False]:
                #     for custom_r in [True, False]:
                a = ActorCriticFetch(env, raw=False, concat=False)
                model_save = f'negLoss-fetchReach-plr:{plr}-vlr:{vlr}-raw:{False}-customr:{True}.h5'

                print("""{}""".format(model_save))
                a.load_weights('init_weights.h5')
                a.train(p_lr=plr,
                        v_lr=vlr,
                        num_eps=500,
                        custom_r=True,
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
        a.train(p_lr=1e-5,
                num_eps=args.epochs,
                render=args.render,
                model_start=args.model,
                custom_r=False)



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
