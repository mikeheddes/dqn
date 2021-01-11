import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gym
import random
import numpy as np
from replay_memory import ReplayMemory
from q_network import QNetwork


MAX_NUM_FRAMES = 200_000
MAX_EPISODE_DURATION = 100  # number of frames
BATCH_SIZE = 24
REWARD_DISCOUNT = 0.9
FRAMES_PER_PRINT = 100
FRAMES_PER_SAVE = 500


def take_random_action(frame):
    probability = 1.0 - frame / (MAX_NUM_FRAMES / 10)
    probability = max(0.1, probability)
    return probability > random.random()


def make_training_transformer(state_shape, agent):
    # create variable before to minimize memory copies
    state_batch_shape = (BATCH_SIZE, *state_shape)
    state_batch = np.zeros(state_batch_shape, dtype=np.float32)
    action_batch = np.zeros((BATCH_SIZE,), dtype=np.int)
    reward_batch = np.zeros((BATCH_SIZE,), dtype=np.float32)
    next_state_batch = np.zeros(state_batch_shape, dtype=np.float32)
    done_batch = np.zeros((BATCH_SIZE,), dtype=np.bool)
    batch_range = np.arange(BATCH_SIZE, dtype=np.int)

    def get_training_batch(experience_samples):
        for i in range(BATCH_SIZE):
            state, action, reward, next_state, done = experience_samples[i]
            state_batch[i] = state
            action_batch[i] = action
            reward_batch[i] = reward
            next_state_batch[i] = next_state
            done_batch[i] = done

        np.invert(done_batch, out=done_batch)
        qs = agent.get_q_values(state_batch, BATCH_SIZE)
        next_max_q = agent.get_q_values(next_state_batch, BATCH_SIZE).max(-1)
        target_batch = reward_batch + REWARD_DISCOUNT * next_max_q * done_batch
        qs[batch_range, action_batch] = target_batch

        return state_batch, qs

    return get_training_batch


def main():
    env = gym.make('CartPole-v1')
    replay_memory = ReplayMemory()
    agent = QNetwork(env.observation_space, env.action_space)
    get_training_batch = make_training_transformer(
        env.observation_space.shape, agent)

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
                # on done doesn't return a negative reward...
                reward *= -1

            experience = (state, action, reward, next_state, done)
            replay_memory.append(experience)
            frame += 1

            experience_samples = replay_memory.sample(BATCH_SIZE)
            state_batch, qs_batch = get_training_batch(experience_samples)
            acc_state_value += np.mean(qs_batch)

            loss = agent.training_step(state_batch, qs_batch)
            acc_loss += loss

            if frame % FRAMES_PER_SAVE == 0:
                model_filename = f"ckpt-loss={loss:.4f}"
                agent.save_model(model_filename)

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
