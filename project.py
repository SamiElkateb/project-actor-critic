import argparse
import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ale_py import ALEInterface
from ale_py.roms import Pong
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers.legacy import RMSprop

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_MODEL_VERSION = "v17"
MODEL_PATH = os.path.join(CURRENT_DIR, "models", CURRENT_MODEL_VERSION)
ACTOR_PATH = os.path.join(MODEL_PATH, "actor.h5")
CRITIC_PATH = os.path.join(MODEL_PATH, "critic.h5")
REWARDS_PATH = os.path.join(MODEL_PATH, "rewards.csv")
DEBUG_PATH = os.path.join(CURRENT_DIR, "debug")
IS_DEBUG = False

NO_ACTION = 0
UP_ACTION = 2
DOWN_ACTION = 3
ACTIONS = [UP_ACTION, DOWN_ACTION]
WIDTH = 80
HEIGHT = 80
SKIP_GRAPHS = [1, 2, 3, 4, 5, 7, 8, 10, 11]
GRAPH_LEGENDS = {
    "0-initial": "Initial Model Dense(512)",
    "1-conv2d-project": "Conv2D Pong Project",
    "2-conv2D-article": "Conv2D Article",
    "3-reward-shaping": "Dense(512) + Reward Shaping",
}

AGENT_MODEL_MAPPING = {
    "initial-model": "0-initial",
    "pong-project": "1-conv2d-project",
    "article-model": "2-conv2D-article",
    "reward-shaping": "3-reward-shaping",
}

WIN_REWARD = 2
HIT_BALL_REWARD = 1
LOSS_REWARD = -1

loaded_rewards = pd.DataFrame({"reward_sum": []})

actor = Sequential(
    [
        Dense(512, activation="elu", input_shape=(2 * HEIGHT * WIDTH,)),
        Dense(len(ACTIONS), activation="softmax"),
    ]
)
critic = Sequential(
    [
        Dense(512, activation="elu", input_shape=(2 * HEIGHT * WIDTH,)),
        Dense(1),
    ]
)
print(actor.summary())

actor.compile(optimizer=RMSprop(1e-4), loss="sparse_categorical_crossentropy")
critic.compile(optimizer=RMSprop(1e-4), loss="mse")

gamma = 0.99


def plot(reward_sums):
    SIZE = 300
    data = {}
    for i, reward_sum in reward_sums.items():
        arr = (
            reward_sum.rolling(window=50)
            .mean()
            .dropna()
            .truncate(after=SIZE)
            .to_numpy()
            .flatten()
        )
        padded_arr = np.pad(
            arr, (0, SIZE - len(arr)), mode="constant", constant_values=np.NaN
        )
        data[i] = padded_arr

    pd.DataFrame(data).plot()
    plt.xlabel("Nombre d'épisode")
    plt.ylabel("Moyenne mobile des gains / épisode")
    plt.savefig("rolling_average_graph.png")
    plt.show()


def stats(mode):
    existing_stats = {}
    for model_dir, legend in GRAPH_LEGENDS.items():
        filepath = os.path.join(CURRENT_DIR, "models", model_dir, "rewards.csv")
        if os.path.exists(filepath):
            existing_stats[legend] = pd.read_csv(filepath)

    for i, stat in existing_stats.items():
        print(f"v{i} mean", stat.reward_sum.mean())

    if mode == "stats":
        plot(existing_stats)


def discount_rewards(r):
    """take 1D float array of rewards and compute discounted reward. http://karpathy.github.io/2016/05/31/rl/"""
    discounted_r = np.zeros((len(r),))
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] == WIN_REWARD or r[t] == LOSS_REWARD:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def compute_hit_ball_bonus(obs_t, obs_tp1):
    if type(obs_t) != np.ndarray or type(obs_tp1) != np.ndarray:
        return 0
    crop_obs_t = ((obs_t[34:194:4, 40:142:2, 2] > 50).astype(np.uint8)).astype(float)
    crop_obs_tp1 = ((obs_tp1[34:194:4, 40:142:2, 2] > 50).astype(np.uint8)).astype(
        float
    )
    ball_t = np.argwhere(crop_obs_t[:, :-1] == 1)
    ball_tp1 = np.argwhere(crop_obs_tp1[:, :-1] == 1)

    if len(ball_t) < 1 or len(ball_tp1) < 1:
        return 0
    has_hit_ball = ball_t[0][1] == 49 and ball_tp1[0][1] == 47
    return 1 if has_hit_ball else 0


def crop(obs):
    """prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. http://karpathy.github.io/2016/05/31/rl/"""
    if obs is None or type(obs) != np.ndarray:
        return np.zeros((HEIGHT * WIDTH))
    obs = obs[35:195]
    obs = obs[::2, ::2, 0]
    obs[obs == 144] = 0
    obs[obs == 109] = 0
    obs[obs != 0] = 1
    return obs.astype(float).ravel()


def train_actor_critic():
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.reset()

    reward_sums = []

    for ep in range(2000):
        Xs, ys, rewards, mod_rewards = [], [], [], []
        prev_obs, obs = None, env.reset()
        for t in range(99000):
            x = np.hstack([crop(obs), crop(prev_obs)])

            hit_ball_bonus = compute_hit_ball_bonus(prev_obs, obs)
            prev_obs = obs
            action_probs = actor.predict(x[None, :], verbose=0)
            ya = np.random.choice(len(ACTIONS), p=action_probs[0])
            action = ACTIONS[ya]

            obs, reward, done, *_ = env.step(action)
            reward = WIN_REWARD if reward == 1 else reward

            Xs.append(x)
            ys.append(ya)
            rewards.append(reward)
            mod_reward = reward + hit_ball_bonus
            mod_rewards.append(mod_reward)

            if done:
                Xs = np.array(Xs)
                ys = np.array(ys)
                values = critic.predict(Xs, verbose=0)[:, 0]
                discounted_rewards = discount_rewards(mod_rewards)
                advantages = discounted_rewards - values
                print(f"adv: {np.min(advantages):.2f}, {np.max(advantages):.2f}")

                actor.fit(Xs, ys, sample_weight=advantages, epochs=1, batch_size=1024)
                critic.fit(Xs, discounted_rewards, epochs=1, batch_size=1024)
                reward_sum = sum(rewards)
                reward_sums.append(reward_sum)
                avg_reward_sum = sum(reward_sums[-50:]) / len(reward_sums[-50:])
                mod_reward_sum = sum(mod_rewards)

                print(
                    f"Episode {ep} -- reward_sum: {reward_sum}, avg_reward_sum: {avg_reward_sum}\n"
                )
                print(f"Episode {ep} -- mod_reward_sum: {mod_reward_sum}\n")

                if ep % 2 == 0:
                    current_df = pd.DataFrame({"reward_sum": reward_sums})
                    df_to_save = pd.concat([loaded_rewards, current_df])
                    df_to_save.to_csv(REWARDS_PATH, index=False)
                    actor.save_weights(ACTOR_PATH)
                    critic.save_weights(CRITIC_PATH)
                break


def play_neural_net(agent):
    env = gym.make("ALE/Pong-v5", render_mode="human")
    model_path = os.path.join(
        CURRENT_DIR, "models", AGENT_MODEL_MAPPING[agent], "actor_model.h5"
    )

    actor = load_model(model_path)

    reward_sum = 0
    prev_obs, obs = None, env.reset()
    env.render()
    X = None
    for t in range(99000):
        x = np.hstack([crop(obs), crop(prev_obs)])
        prev_obs = obs
        if agent == "pong-project" or agent == "article-model":
            X = x.reshape((-1, 80, 160, 1))
        else:
            X = x[None, :]

        action_probs = actor.predict(X, verbose=0)
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
        choices=["train", "test", "stats"],
        help="""
        Accepted values: train | test | stats.
        The mode of the python script.
        The train mode is for training the agent.
        The test mode is for watching the agent play.
        The stats mode to see the stats graph.
        """,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Will create images of the observation state in the debug folder.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="reward-shaping",
        choices=["initial-model", "pong-project", "article-model", "reward-shaping"],
        help="""The algorithm that train the model that the agent will use""",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    stats(args.mode)

    IS_DEBUG = args.debug
    if IS_DEBUG:
        os.makedirs(DEBUG_PATH, exist_ok=True)

    if args.mode == "stats":
        pass
    elif args.mode == "test":
        print(f"Starting the script in {args.mode} mode with ...")
        ale = ALEInterface()
        ale.loadROM(Pong)
        play_neural_net(args.agent)
    else:
        ale = ALEInterface()
        ale.loadROM(Pong)
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
