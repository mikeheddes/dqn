import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CHECKPOINT_DIR = os.path.abspath("./ckpt")
LOG_DIR = os.path.abspath("./log")


def mean_value(y_true, y_pred):
    return tf.reduce_mean(y_pred)


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
        checkpoints = [CHECKPOINT_DIR + "/" +
                       name for name in os.listdir(CHECKPOINT_DIR)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            return keras.models.load_model(latest_checkpoint, 
            # custom_objects={"mean_value": mean_value}
            )

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

        model.compile(
            # Optimizer
            optimizer=keras.optimizers.RMSprop(),
            # Loss function to minimize
            loss=keras.losses.MeanSquaredError(),
            # List of metrics to monitor
            # metrics=[mean_value],
        )

        return model

    def perdict(self, x, batch_size=None):
        return self.model.predict(x, batch_size=batch_size)

    def act(self, state):
        state = state.reshape((1, -1))
        return np.argmax(self.model.predict(state))

    def training_step(self, x, y, batch_size, frame):

        callbacks = [
            # We include the training loss in the saved model name.
            keras.callbacks.ModelCheckpoint(
                filepath=CHECKPOINT_DIR + "/ckpt-loss={loss:.4f}"),

            # keras.callbacks.TensorBoard(
            #     log_dir=LOG_DIR,
            #     histogram_freq=0,  # How often to log histogram visualizations
            #     embeddings_freq=0,  # How often to log embedding visualizations
            #     update_freq="epoch",
            # )  # How often to write logs (default: once per epoch)
        ]

        if frame % 100 != 0:
            callbacks = None

        self.model.fit(x=x, y=y, batch_size=batch_size,
                       epochs=1, callbacks=callbacks)
