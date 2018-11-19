import numpy as np
from collections import deque

from tqdm import tqdm
# from ddpg import Actor
# from ddpg import Critic
from actor import Actor
from critic import Critic

import random
import tensorflow as tf
import gym
from gym.envs.robotics.fetch.reach import FetchReachEnv
from gym.envs.robotics.fetch_env import goal_distance
import argparse
import tensorflow.contrib.eager as tfe

# from ddpg import Environment
# from ddpg import DDPG

tf.enable_eager_execution()
# from utils.memory_buffer import MemoryBuffer

class OrnsteinUhlenbeckActionNoise:
    """
    From OpenAI's implementation of Ornstein Action Noise
    """
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class DDPG():
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, act_range, k, buffer_size=10000, gamma=0.99, lr=0.001, tau=0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.env_dim = (1,) + (13,)
        self.gamma = gamma
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, act_dim, lr, tau)
        # self.buffer = MemoryBuffer(buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.count = 0
        self.buffer_size = buffer_size

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        experience = (state, action, reward, done, new_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch, a_batch, r_batch, d_batch, s2_batch = [], [], [], [], []
        for s_, a_, r_, d_, s2_ in batch:
            s_batch.append(s_)
            s2_batch.append(s2_)
            a_batch.append(a_)
            r_batch.append(r_)
            d_batch.append(d_)
        s_batch = np.squeeze(np.array(s_batch), axis=1)
        s2_batch = np.squeeze(np.array(s2_batch), axis=1)
        r_batch = np.reshape(np.array(r_batch), (len(r_batch), 1))
        a_batch = np.array(a_batch)

        d_batch = np.reshape(np.array(d_batch, dtype=int), (len(batch), 1))
        return s_batch, a_batch, r_batch, d_batch, s2_batch

        # # Return a batch of experience
        # s_batch = np.array([i[0] for i in batch])
        # a_batch = np.array([i[1] for i in batch])
        # r_batch = np.array([i[2] for i in batch])
        # d_batch = np.array([i[3] for i in batch])
        # new_s_batch = np.array([i[4] for i in batch])
        # return s_batch, a_batch, r_batch, d_batch, new_s_batch

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model(states)
        grads = self.critic.gradients(states, tfe.Variable(actions))
        # Train actor
        self.actor.train(states, grads)
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def format_state(self, state):
        ob_1 = np.reshape(state['observation'], (1, 10))
        de_1 = np.reshape(state['desired_goal'], (1, 3))
        return np.concatenate([ob_1, de_1], axis=1)

    def store_states(self, state, action, reward, done, info, new_state):
        # print(state['observation'].shape)
        ob_1 = np.reshape(state['observation'], (1, 10))
        ac_1 = np.reshape(state['achieved_goal'], (1, 3))
        de_1 = np.reshape(state['desired_goal'], (1, 3))
        ob_2 = np.reshape(new_state['observation'], (1, 10))
        s_1 = np.concatenate([ob_1, ac_1], axis=1)
        s2_1 = np.concatenate([ob_2, ac_1], axis=1)
        s_2 = np.concatenate([ob_1, de_1], axis=1)
        s2_2 = np.concatenate([ob_2, de_1], axis=1)
        substitute_goal = state['achieved_goal'].copy()
        substitute_reward = env.compute_reward(state['achieved_goal'], substitute_goal, info)

        e1 = (s_2, action, reward, done, s2_2)
        e2 = (s_1, action, substitute_reward, True, s2_1)
        if self.count + 2 < self.buffer_size:
            self.count += 2
        else:
            self.buffer.popleft()
            self.buffer.popleft()
        self.buffer.append(e1)
        self.buffer.append(e2)

    def train(self, env, args):
        results = []
        num_steps = 200
        # First, gather experience
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit="episode")

        noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_dim))
        avg_r_ep = 0

        best_avg = 0
        best_score = -float('inf')

        past_samples = 15
        hist_ratio = deque(maxlen=past_samples)
        hist_scores = deque(maxlen=past_samples)
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            s = env.reset()
            # noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

            for _ in range(num_steps):
                if args.render: env.render()
                # Actor picks an action (following the deterministic policy)
                old_state = self.format_state(s)
                # print(old_state.shape)
                a = self.policy_action(old_state)
                # Clip continuous values to be valid w.r.t. environment
                a = np.clip(a + noise(), -self.act_range, self.act_range)
                # Retrieve new state, reward, and whether the state is terminal
                a = np.squeeze(a)
                new_state, r, done, info = env.step(a)
                dist = goal_distance(new_state['achieved_goal'], new_state['desired_goal'])
                # new_state = new_state['observation']

                # Add outputs to memory buffer
                # self.memorize(old_state, a, r, done, new_state)
                self.store_states(s, a, r, done, info, new_state)

                s = new_state
                cumul_reward += r

                if args.batch_size < self.count:
                    # Sample experience from buffer
                    states, actions, rewards, dones, new_states = self.sample_batch(args.batch_size)
                    # Predict target q-values using target networks
                    # print(new_states.shape)
                    # print(self.actor.target_predict(new_states).shape)
                    q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
                    # Compute critic target
                    critic_target = self.bellman(rewards, q_values, dones)
                    # Train both networks on sampled batch, update target networks
                    self.update_models(states, actions, critic_target)
                    # Update current state

                if done:
                    break

            if avg_r_ep == 0:
                avg_r_ep = cumul_reward
            else:
                avg_r_ep = avg_r_ep * 0.99 + cumul_reward * 0.01

            if avg_r_ep > best_avg:
                best_avg = avg_r_ep
                self.actor.model.save_weights('pretrained/best_avg_ddpgActor.h5')
                self.critic.model.save_weights('pretrained/best_avg_ddpgCritic.h5')
            # Display score
            if cumul_reward >= best_score:
                best_score = cumul_reward
                self.actor.model.save_weights('pretrained/ddpgActor.h5')
                self.critic.model.save_weights('pretrained/ddpgCrtic.h5')

            hist_ratio.append(int(dist <= 0.05))
            hist_scores.append(cumul_reward)

            tqdm_e.set_description("Score: {} | "
                                   "Best Reward: {} | "
                                   "Avg Reward, solve ratio over last {} samples: {:.3f}, {:.3f}".format(cumul_reward,
                                                                                                        best_score,
                                                                                                        past_samples,
                                                                                                        np.mean(hist_scores),
                                                                                                        np.mean(hist_ratio)
                                                                                                        ))
            tqdm_e.refresh()

        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size (experience replay)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    args = parser.parse_args()

    consec_frames = 4
    # Continuous Environments Wrapp
    # er
    # env = gym.make('FetchReach-v1')
    env = FetchReachEnv(reward_type='sparse')
    env.reset()

    state_dim = env.reset()['observation'].shape
    action_space = FetchReachEnv().action_space
    action_dim = action_space.high.shape[0]
    act_range = action_space.high

    algo = DDPG(action_dim, state_dim, act_range, consec_frames)
    algo.train(env, args)