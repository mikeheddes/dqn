import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CHECKPOINT_DIR = os.path.abspath("./ckpt")
LOG_DIR = os.path.abspath("./log")


class Q_Network:
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
        model = keras.Sequential(
            [
                keras.Input(shape=self.state_space.shape),
                layers.Dense(24, activation="relu"),
                layers.Dense(24, activation="relu"),
                layers.Dense(self.action_space.n),
            ]
        )

        model.compile('rmsprop', 'mse')
        return model

    def get_q_values(self, x, batch_size=None):
        return self.model.predict(x, batch_size=batch_size)

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        return np.argmax(self.model.predict(state), axis=-1)[0]

    def training_step(self, x, y, batch_size, frame):
        loss = self.model.train_on_batch(x=x, y=y)

        if frame % 500 == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"ckpt-loss={loss:.4f}")
            self.model.save(save_path)

        return loss
