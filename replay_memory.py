from collections import deque
import random
import operator


class ReplayMemory:
    def __init__(self, maxlen=10_000):
        self.data = deque([], maxlen=maxlen)

    def append(self, experience):
        self.data.append(experience)

    def sample(self, num=1):
        len_experience = len(self.data)

        if len_experience == 0:
            raise RuntimeError("Replay Memory contains no experiences")

        indices = [None] * num
        for i in range(num):
            indices[i] = random.randrange(len_experience)

        getter = operator.itemgetter(*indices)
        return getter(self.data)
