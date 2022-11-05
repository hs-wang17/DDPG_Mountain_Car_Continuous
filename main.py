# --------------------------------------
# Main
# Author: Wang Huisheng
# Date: 2022.11.15
# --------------------------------------

import tensorflow.compat.v1 as tf
import numpy as np
import gym
from replay_buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from ou_noise import OUNoise
import imageio

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Soft target update param
TAU = 0.001
RANDOM_SEED = 0
DECREASE_EXPLORE = 1 - 1 / 70
DEVICE = '/gpu:0'


def train(epochs=100, MINIBATCH_SIZE=40, GAMMA=0.99, epsilon=1.0, min_epsilon=0.01, BUFFER_SIZE=10000,
          render=False):
    with tf.Session() as sess:
        env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = np.float64(10)

        ruido = OUNoise(action_dim, mu=0.4)  # Ornstein-Uhlenbeck Noise
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU, DEVICE)
        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), DEVICE)

        sess.run(tf.global_variables_initializer())
        actor.update_target_network()
        critic.update_target_network()
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

        goal = 0
        max_state = -1.

        # Load models
        # critic.recover_critic()
        # actor.recover_actor()

        for i in range(epochs):
            # Record
            if render:
                video = imageio.get_writer('./output_images/train_%d.gif' % i, fps=30)

            state, info = env.reset()
            state = np.hstack(state)
            ep_reward = 0
            ep_ave_max_q = 0
            done = False
            step = 0
            max_state_episode = -1
            # Decrease epsilon
            epsilon *= DECREASE_EXPLORE
            epsilon = max(min_epsilon, epsilon)

            while not done:
                action = actor.predict(np.reshape(state, (1, state_dim)))
                action = action + max(epsilon, 0) * ruido.noise()
                next_state, reward, done, truncated, info = env.step(action)
                replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                                  done, np.reshape(next_state, (actor.s_dim,)))
                if replay_buffer.size() > MINIBATCH_SIZE:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                    y_i = []
                    for k in range(MINIBATCH_SIZE):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + GAMMA * target_q[k])
                    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
                    ep_ave_max_q += max(predicted_q_value)
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])
                    actor.update_target_network()
                    critic.update_target_network()
                state = next_state
                if next_state[0] > max_state_episode:
                    max_state_episode = next_state[0]
                ep_reward = ep_reward + reward
                step += 1

                if render:
                    video.append_data(env.render())

            if done:
                ruido.reset()
                if state[0] > 0.45:
                    goal += 1
            if max_state_episode > max_state:
                max_state = max_state_episode

            print('th', i + 1,
                  'n steps', step,
                  'R:', round(ep_reward, 3),
                  'Pos', round(epsilon, 3),
                  'Efficiency', round(100. * (goal / (i + 1.)), 3))

        critic.save_critic()
        actor.save_actor()


def test():
    with tf.Session() as sess:
        env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = np.float64(10)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU, DEVICE)
        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), DEVICE)

        # Load models
        actor.recover_actor()
        critic.recover_critic()

        # Record video
        video = imageio.get_writer('./output_images/test.gif', fps=30)

        state, info = env.reset()
        state = np.hstack(state)
        done = False
        while not done:
            action = actor.predict(np.reshape(state, (1, state_dim)))
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            video.append_data(env.render())


if __name__ == '__main__':
    train(render=True)
    test()
