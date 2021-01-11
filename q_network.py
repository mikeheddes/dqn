import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CHECKPOINT_DIR = os.path.abspath("./ckpt")
LOG_DIR = os.path.abspath("./logs")


class QNetwork:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

        # Prepare a directory to store all the checkpoints.
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        self.model = self.make_or_restore_model()

    def make_or_restore_model(self):
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
        checkpoints = [os.path.join(CHECKPOINT_DIR, name)
                       for name in os.listdir(CHECKPOINT_DIR)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            return keras.models.load_model(latest_checkpoint)

        print("Creating a new model")
        return self.get_compiled_model()

    def get_compiled_model(self):
        Lin = keras.Input(shape=self.state_space.shape)
        L1 = layers.Dense(24, activation="relu")
        L2 = layers.Dense(24, activation="relu")
        Lout = layers.Dense(self.action_space.n)

        model = keras.Sequential([Lin, L1, L2, Lout])
        model.compile('rmsprop', 'mse')
        return model

    def get_q_values(self, x, batch_size=None):
        return self.model.predict(x, batch_size=batch_size)

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        return np.argmax(self.model.predict(state), axis=-1)[0]

    def training_step(self, x, y):
        loss = self.model.train_on_batch(x=x, y=y)
        return loss

    def save_model(self, filename):
        save_path = os.path.join(CHECKPOINT_DIR, filename)
        self.model.save(save_path)
