import argparse
import os
import sys

import cv2
import gymnasium as gym
import pandas as pd
from ale_py import ALEInterface
from ale_py.roms import Pong
from matplotlib.pyplot import np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers.legacy import RMSprop

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_MODEL_VERSION = "v3"
MODEL_PATH = os.path.join(CURRENT_DIR, "models", CURRENT_MODEL_VERSION)
ACTOR_PATH = os.path.join(MODEL_PATH, "actor.h5")
CRITIC_PATH = os.path.join(MODEL_PATH, "critic.h5")
REWARDS_PATH = os.path.join(MODEL_PATH, "rewards.csv")
DEBUG_PATH = os.path.join(CURRENT_DIR, "debug")
IS_DEBUG = False

UP_ACTION = 2
DOWN_ACTION = 3
ACTIONS = [UP_ACTION, DOWN_ACTION]

loaded_rewards = pd.DataFrame({"reward_sum": []})

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


def prepro(I):
    """prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. http://karpathy.github.io/2016/05/31/rl/"""
    if I is None or type(I) != np.ndarray:
        return np.zeros((6400,))
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(float).ravel()


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


class Observation:
    # Pour obtenir une observation
    def __init__(self, obs_t, obs_tp1) -> None:
        self.action = None
        self.obs = Observation.__crop__(obs_t)
        self.obs_tp1 = Observation.__crop__(obs_tp1)

    @staticmethod
    def __crop__(obs):
        if obs is None or type(obs) != np.ndarray:
            return np.zeros(40 * 51).reshape(40, 51).ravel()
        # On coupe l'image pour ne garder que la partie intéressante du jeu,
        # sans le score, la raquette de l'ennemi et les bandes sur les cotés de l'écran
        return (
            ((obs[34:194:4, 40:142:2, 2] > 50).astype(np.uint8)).astype(float).ravel()
        )

    debug_image_nb = 0

    def get_obs(self):
        state_before_copy = self.obs.copy()
        state_before_copy[:, -1] = 0
        return (self.obs_tp1 - state_before_copy).reshape(1, 40 * 51)


def train_actor_critic():
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.reset()

    reward_sums = []
    for ep in range(2000):
        Xs, ys, rewards = [], [], []
        prev_obs, obs = None, env.reset()
        for t in range(99000):
            x = np.hstack([prepro(obs), prepro(prev_obs)])
            prev_obs = obs

            action_probs = actor.predict(x[None, :], verbose=0)
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

                if ep % 10 == 0:
                    current_df = pd.DataFrame({"reward_sum": reward_sums})
                    df_to_save = pd.concat([loaded_rewards, current_df])
                    df_to_save.to_csv(REWARDS_PATH, index=False)
                    actor.save_weights(ACTOR_PATH)
                    critic.save_weights(CRITIC_PATH)
                break


def play_neural_net():
    env = gym.make("ALE/Pong-v5", render_mode="human")
    actor.load_weights(ACTOR_PATH)

    reward_sum = 0
    prev_obs, obs = None, env.reset()
    env.render()
    for t in range(99000):
        x = np.hstack([Observation.__crop__(obs), Observation.__crop__(prev_obs)])
        prev_obs = obs

        action_probs = actor.predict(x, verbose=0)
        ya = np.random.choice(len(ACTIONS), p=action_probs[0])
        action = ACTIONS[ya]

        obs, reward, done, *_ = env.step(action)
        reward_sum += reward

        if reward != 0:
            print(f"t: {t} -- reward: {reward}")

        if done:
            print(f"t: {t} -- reward_sum: {reward_sum}")
            break


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

    v0_data = pd.read_csv(os.path.join(CURRENT_DIR, "models", "v0", "rewards.csv"))
    v1_data = pd.read_csv(os.path.join(CURRENT_DIR, "models", "v1", "rewards.csv"))
    v2_data = pd.read_csv(os.path.join(CURRENT_DIR, "models", "v2", "rewards.csv"))
    v3_data = pd.read_csv(os.path.join(CURRENT_DIR, "models", "v3", "rewards.csv"))

    print("v0 mean", v0_data.reward_sum.mean())
    print("v1 mean", v1_data.reward_sum.mean())
    print("v2 mean", v2_data.reward_sum.mean())
    print("v3 mean", v3_data.reward_sum.mean())

    IS_DEBUG = args.debug
    if IS_DEBUG:
        os.makedirs(DEBUG_PATH, exist_ok=True)

    if args.mode == "watch":
        print(f"Starting the script in {args.mode} mode with ...")
        play_neural_net()
    else:
        print(f"Starting the script in {args.mode} mode ...")
        os.makedirs(MODEL_PATH, exist_ok=True)

        if (
            os.path.exists(ACTOR_PATH)
            and os.path.exists(CRITIC_PATH)
            and os.path.exists(REWARDS_PATH)
        ):
            print(f"Loading model {CURRENT_MODEL_VERSION}")
            actor.load_weights(ACTOR_PATH)
            critic.load_weights(CRITIC_PATH)
            loaded_rewards = pd.read_csv(REWARDS_PATH)
            train_actor_critic()
        elif (
            not os.path.exists(ACTOR_PATH)
            and not os.path.exists(CRITIC_PATH)
            and not os.path.exists(REWARDS_PATH)
        ):
            print(f"Creating model {CURRENT_MODEL_VERSION}")
            train_actor_critic()
        else:
            raise Exception("Either actor or critic model exists and not its sibling")
