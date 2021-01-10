import gym
import random
import numpy as np
from replay_memory import ReplayMemory
from qnetwork import QNetwork

MAX_NUM_FRAMES = 1_000
MAX_EPISODE_DURATION = 100  # number of frames

def main():
    env = gym.make('CartPole-v1')
    agent = QNetwork(env.observation_space, env.action_space)

    frame = 0
    while frame < MAX_NUM_FRAMES:
        state = env.reset()
        for t in range(MAX_EPISODE_DURATION):
            env.render()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            frame = frame + 1

            if done or frame == MAX_NUM_FRAMES:
                break

            state = next_state

    env.close()


if __name__ == "__main__":
    main()
