import argparse
import os
import sys

import cv2
import gymnasium as gym
import pandas as pd
from ale_py import ALEInterface
from ale_py.roms import Pong
from gymnasium.utils.play import play
from joblib import load
from keras.src.saving.saving_api import load_model
from matplotlib.pyplot import Enum, np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers.legacy import RMSprop

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data")
X_PATH = os.path.join(DATA_PATH, "X.csv")
Y_PATH = os.path.join(DATA_PATH, "y.csv")
NEURAL_NET_PATH = os.path.join(CURRENT_DIR, "models", "pong-smoteen.h5")
RANDOM_FOREST_PATH = os.path.join(CURRENT_DIR, "models", "pong-random-forest.joblib")
DEBUG_PATH = os.path.join(CURRENT_DIR, "debug")
IS_DEBUG = False

UP_ACTION = 2
DOWN_ACTION = 3
ACTIONS = [UP_ACTION, DOWN_ACTION]

loaded_df = pd.DataFrame({"reward_sum": []})
try:
    loaded_df = pd.read_csv("reinforcement-learning/models/actor-critic/v3/rewards.csv")
except Exception:
    pass

# Neural net model takes the state and outputs action and value for that state
actor = Sequential(
    [
        Dense(512, activation="elu", input_shape=(2 * 6400,)),
        Dense(len(ACTIONS), activation="softmax"),
    ]
)
critic = Sequential([Dense(512, activation="elu", input_shape=(2 * 6400,)), Dense(1)])

actor.compile(optimizer=RMSprop(1e-4), loss="sparse_categorical_crossentropy")
critic.compile(optimizer=RMSprop(1e-4), loss="mse")

gamma = 0.99


def discount_rewards(r):
    discounted_r = np.zeros((len(r),))
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = (
                0  # reset the sum, since this was a game boundary (pong specific!)
            )
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class PongActions(Enum):
    NO_ACTION = 0
    UP = 2
    DOWN = 3

    @staticmethod
    def from_sparse_categorical(num):
        if num == 0:
            return PongActions.NO_ACTION.value
        if num == 1:
            return PongActions.UP.value
        if num == 2:
            return PongActions.DOWN.value
        raise ValueError("Invalid Pong Action Category")

    def to_sparse_categorical(self):
        if self == PongActions.NO_ACTION:
            return 0
        if self == PongActions.UP:
            return 1
        if self == PongActions.DOWN:
            return 2
        raise ValueError("Invalid Pong Action Category")


class Observation:
    # Pour obtenir une observation
    def __init__(self, obs_t, obs_tp1) -> None:
        self.action = None
        self.obs = Observation.__crop__(
            obs_t
        )  # On decoupe l'image pour ne garder que la partie interessante,
        self.obs_tp1 = Observation.__crop__(
            obs_tp1
        )  # en noir et blanc pour reduire les dimensions

        # On identifie la position de la balle sur l'image,
        # ce qui permet de choisir les images que l'on souhaite sauvegarder
        obs_t_ball_only = self.obs.copy()[:, :-1]
        ball_t = np.argwhere(obs_t_ball_only == 1)
        obs_tp1_ball_only = self.obs_tp1.copy()[:, :-1]
        ball_tp1 = np.argwhere(obs_tp1_ball_only == 1)

        self.is_ball_on_field = len(ball_t) > 0 or len(ball_tp1) > 0

        if len(ball_t) > 0 and len(ball_tp1) > 0:
            self.is_ball_going_towards_enemy = ball_t[0][1] > ball_tp1[0][1]
        else:
            self.is_ball_going_towards_enemy = False

    def add_action(self, action):
        self.action = [PongActions(action).to_sparse_categorical()]

    def save(self):
        with open(
            X_PATH, "a"
        ) as outfile_X:  # On sauvegarde la difference entre l'etat actuel et l'etat suivant
            # pour avoir une indication sur la direction de la balle lors de la prediction
            state_before_copy = self.obs.copy()
            state_before_copy[:, -1] = 0  # Si la raquette ne bouge pas, la soustraction
            # des deux images la ferait disparaitre de l'observation
            # On met donc a zero la colonne correspondant a la raquette dans l'état actuel pour corriger ce problème
            diff = self.obs_tp1 - state_before_copy
            np.savetxt(outfile_X, delimiter=",", X=[diff.flatten()], fmt="%d")
            if IS_DEBUG:
                Observation.save_debug(diff)
        with open(Y_PATH, "a") as outfile_Y:
            np.savetxt(outfile_Y, delimiter=",", X=self.action, fmt="%d")

    @staticmethod
    def __crop__(obs):
        # On coupe l'image pour ne garder que la partie intéressante du jeu,
        # sans le score, la raquette de l'ennemi et les bandes sur les cotés de l'écran
        return ((obs[34:194:4, 40:142:2, 2] > 50).astype(np.uint8)).astype(float)

    debug_image_nb = 0

    @staticmethod
    def save_debug(input_obs):
        obs = input_obs if len(input_obs.shape) == 2 else input_obs[0, :, :, 0]
        rgb_image = np.zeros((obs.shape[0], obs.shape[1], 3), dtype=np.uint8)
        rgb_image[obs == -1, 0] = 255
        rgb_image[obs == 1, 2] = 255
        filename = os.path.join(DEBUG_PATH, f"image_{Observation.debug_image_nb}.png")
        cv2.imwrite(filename, rgb_image)
        Observation.debug_image_nb += 1

    @staticmethod
    def preprocess_obs(obs):
        return (Observation.__crop__(obs)).reshape(-1, 40, 51, 1)

    def get_obs(self):
        state_before_copy = self.obs.copy()
        state_before_copy[:, -1] = 0
        return self.obs_tp1 - state_before_copy


def train_actor_critic():
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.reset()

    reward_sums = []
    for ep in range(2000):
        Xs, ys, rewards = [], [], []
        prev_obs, obs = None, env.reset()
        for t in range(99000):
            # x = np.hstack([Observation.preprocess_obs(obs), prepro(prev_obs)])
            x = Observation(prev_obs, obs).get_obs()
            prev_obs = obs

            action_probs = actor.predict(x, verbose=0)
            ya = np.random.choice(len(ACTIONS), p=action_probs[0])
            action = ACTIONS[ya]

            obs, reward, done, *_ = env.step(action)

            Xs.append(x)
            ys.append(ya)
            rewards.append(reward)

            # if reward != 0: print(f'Episode {ep} -- step: {t}, ya: {ya}, reward: {reward}')

            if done:
                Xs = np.array(Xs)
                ys = np.array(ys)
                values = critic.predict(Xs, verbose=0)[:, 0]
                discounted_rewards = discount_rewards(rewards)
                advantages = discounted_rewards - values
                # advantages = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
                print(f"adv: {np.min(advantages):.2f}, {np.max(advantages):.2f}")

                actor.fit(Xs, ys, sample_weight=advantages, epochs=1, batch_size=1024)
                critic.fit(Xs, discounted_rewards, epochs=1, batch_size=1024)

                reward_sum = sum(rewards)
                reward_sums.append(reward_sum)
                avg_reward_sum = sum(reward_sums[-50:]) / len(reward_sums[-50:])

                print(
                    f"Episode {ep} -- reward_sum: {reward_sum}, avg_reward_sum: {avg_reward_sum}\n"
                )

                if ep % 20 == 0:
                    current_df = pd.DataFrame({"reward_sum": reward_sums})
                    df_to_save = pd.concat([loaded_df, current_df])
                    df_to_save.to_csv(
                        "reinforcement-learning/models/actor-critic/v3/rewards.csv"
                    )
                    actor.save_weights(
                        "reinforcement-learning/models/actor-critic/v3/actor3.h5"
                    )
                    critic.save_weights(
                        "reinforcement-learning/models/actor-critic/v3/critic3.h5"
                    )
                break


def play_neural_net():
    env = gym.make("ALE/Pong-v5", render_mode="human")
    model = load_model(NEURAL_NET_PATH)
    state_before = env.reset()[0]
    state_before = Observation.preprocess_obs(state_before)
    state = None
    env.render()
    while True:
        if state is None or state_before is None:
            action = env.action_space.sample()
            state = state_before
            state_before, *_ = env.step(action)
            state_before = Observation.preprocess_obs(state_before)
            continue

        state[0][:, -1] = 0
        state = state_before - state

        if IS_DEBUG:
            Observation.save_debug(state)

        action = model.predict(state, verbose=False)
        action = np.argmax(action)
        action = PongActions.from_sparse_categorical(action)

        state = state_before
        state_before, *_ = env.step(action)
        state_before = Observation.preprocess_obs(state_before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Pong Game played by a Actor Critic.
                       Authors: EL KATEB Sami, PAUL Thomas"""
    )
    parser.add_argument(
        "mode",
        metavar="mode",
        type=str,
        choices=["play", "watch"],
        help="""
        Accepted values: play | watch.
        The mode of the python script.
        The play mode is for generating data to train the agent.
        The watch mode is for watching the agent play.
        """,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Will create images of the observation state in the debug folder.",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    ale = ALEInterface()
    ale.loadROM(Pong)

    IS_DEBUG = args.debug
    if IS_DEBUG:
        os.makedirs(DEBUG_PATH, exist_ok=True)

    if args.mode == "watch":
        print(f"Starting the script in {args.mode} mode with {args.agent} agent ...")
        play_neural_net()
    else:
        print(f"Starting the script in {args.mode} mode ...")
        os.makedirs(DATA_PATH, exist_ok=True)
        train_actor_critic()
