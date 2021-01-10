import gym
import random
import numpy as np
from replay_memory import ReplayMemory
from qnetwork import QNetwork

MAX_NUM_FRAMES = 10_000
MAX_EPISODE_DURATION = 100  # number of frames
BATCH_SIZE = 24
REWARD_DISCOUNT = 0.9


def take_random_action(frame):
    probability = 1.0 - frame / (MAX_NUM_FRAMES / 10)
    probability = max(0.1, probability)
    return probability > random.random()


def transform_experience(experience_batch, agent):
    states, actions, rewards, next_states, done = list(zip(*experience_batch))
    state_batch = np.array(states)
    action_batch = np.array(actions, dtype=np.int)
    reward_batch = np.array(rewards)
    next_state_batch = np.array(next_states)
    done_batch = np.invert(np.array(done, dtype=np.bool)).astype(np.float32)
    qs = agent.perdict(state_batch, BATCH_SIZE)
    next_max_q = np.max(agent.perdict(next_state_batch, BATCH_SIZE), axis=-1)
    target_batch = reward_batch + REWARD_DISCOUNT * next_max_q * done_batch
    qs[:, action_batch] = target_batch
    return state_batch, qs


def main():
    env = gym.make('CartPole-v1')
    replay_memory = ReplayMemory()
    agent = QNetwork(env.observation_space, env.action_space)

    frame = 0
    while frame < MAX_NUM_FRAMES:
        state = env.reset()
        for t in range(MAX_EPISODE_DURATION):
            if take_random_action(frame):
                action = env.action_space.sample()  # pick random action
            else:
                action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            experience = (state, action, reward, next_state, done)
            replay_memory.append(experience)
            frame = frame + 1

            # learning
            experience_batch = replay_memory.sample(BATCH_SIZE)
            x, y = transform_experience(experience_batch, agent)
            agent.training_step(x, y, BATCH_SIZE, frame)

            if frame % 500 == 0:
                print(f"frame: {frame}")

            if done or frame == MAX_NUM_FRAMES:
                break

            state = next_state

    env.close()


if __name__ == "__main__":
    main()
