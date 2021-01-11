import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from q_network import Q_Network
from replay_memory import ReplayMemory
import numpy as np
import random
import gym


MAX_NUM_FRAMES = 200_000
MAX_EPISODE_DURATION = 100  # number of frames
BATCH_SIZE = 24
REWARD_DISCOUNT = 0.9
FRAMES_PER_PRINT = 100


def take_random_action(frame):
    probability = 1.0 - frame / (MAX_NUM_FRAMES / 10)
    probability = max(0.1, probability)
    return probability > random.random()


def main():
    env = gym.make('CartPole-v1')
    replay_memory = ReplayMemory()
    agent = Q_Network(env.observation_space, env.action_space)

    # create variable before to minimize memory copies
    state_batch_shape = (BATCH_SIZE, *env.observation_space.shape)
    state_batch = np.zeros(state_batch_shape, dtype=np.float32)
    action_batch = np.zeros((BATCH_SIZE,), dtype=np.int)
    reward_batch = np.zeros((BATCH_SIZE,), dtype=np.float32)
    next_state_batch = np.zeros(state_batch_shape, dtype=np.float32)
    done_batch = np.zeros((BATCH_SIZE,), dtype=np.bool)
    batch_range = np.arange(BATCH_SIZE, dtype=np.int)

    frame = 0
    acc_loss = 0
    acc_state_value = 0
    while frame < MAX_NUM_FRAMES:
        state = env.reset()
        for t in range(MAX_EPISODE_DURATION):
            if take_random_action(frame):
                action = env.action_space.sample()  # pick random action
            else:
                action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            if done:
                # done doesn't return a negative reward...
                reward *= -1

            experience = (state, action, reward, next_state, done)
            replay_memory.append(experience)
            frame += 1

            # learning
            experience_batch = replay_memory.sample(BATCH_SIZE)
            for i in range(BATCH_SIZE):
                b_state, b_action, b_reward, b_next_state, b_done = experience_batch[i]
                state_batch[i] = b_state
                action_batch[i] = b_action
                reward_batch[i] = b_reward
                next_state_batch[i] = b_next_state
                done_batch[i] = b_done

            np.invert(done_batch, out=done_batch)
            qs = agent.get_q_values(state_batch, BATCH_SIZE)
            next_max_q = np.max(agent.get_q_values(
                next_state_batch, BATCH_SIZE), axis=-1)
            target_batch = reward_batch + REWARD_DISCOUNT * next_max_q * done_batch
            qs[batch_range, action_batch] = target_batch
            acc_state_value += np.mean(qs)

            loss = agent.training_step(state_batch, qs, BATCH_SIZE, frame)
            acc_loss += loss

            if frame % FRAMES_PER_PRINT == 0:
                print(f"Frame: {frame}")
                avg_loss = acc_loss / FRAMES_PER_PRINT
                avg_state_value = acc_state_value / FRAMES_PER_PRINT
                print(
                    f"avg loss: {avg_loss:.4f}; avg value: {avg_state_value:.2f}")
                acc_loss = 0
                acc_state_value = 0

            if done or frame == MAX_NUM_FRAMES:
                break

            state = next_state

    env.close()


if __name__ == "__main__":
    main()
