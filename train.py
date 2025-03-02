import argparse
from functools import partial
from typing import Dict, Sequence

import numpy as np
import wandb

from dreamerv3_flax.async_vector_env import AsyncVectorEnv
from dreamerv3_flax.buffer import ReplayBuffer
from dreamerv3_flax.env import TASKS, CrafterEnv, VecCrafterEnv
from dreamerv3_flax.jax_agent import JAXAgent


def get_eval_metric(achievements: Sequence[Dict]) -> float:
    achievements = [list(achievement.values()) for achievement in achievements]
    success_rate = 100 * (np.array(achievements) > 0).mean(axis=0)
    score = np.exp(np.mean(np.log(1 + success_rate))) - 1
    eval_metric = {
        "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
        "score": score,
    }
    return eval_metric


def main(args):
    # Logger
    project = "dreamerv3-flax"
    group = f"{args.exp_name}"
    if args.timestamp:
        group += f"-{args.timestamp}"
    name = f"s{args.seed}"
    logger = wandb.init(project=project, group=group, name=name)

    # Seed
    np.random.seed(args.seed)

    # Environment
    env_fns = [partial(CrafterEnv, seed=args.seed)]
    env = VecCrafterEnv(AsyncVectorEnv(env_fns))

    # Buffer
    buffer = ReplayBuffer(env, batch_size=16, num_steps=64)

    # Agent
    agent = JAXAgent(env, seed=args.seed)
    state = agent.initial_state(1)

    # Reset
    actions = env.action_space.sample()
    obs, rewards, dones, firsts, infos = env.step(actions)

    # Train
    achievements = []
    for step in range(1000000):
        actions, state = agent.act(obs, firsts, state)
        buffer.add(obs, actions, rewards, dones, firsts)

        actions = np.argmax(actions, axis=-1)
        obs, rewards, dones, firsts, infos = env.step(actions)
        for done, info in zip(dones, infos):
            if done:
                rollout_metric = {
                    "episode_return": info["episode_return"],
                    "episode_length": info["episode_length"],
                }
                logger.log(rollout_metric, step)
                achievements.append(info["achievements"])
                eval_metric = get_eval_metric(achievements)
                logger.log(eval_metric, step)

        if step >= 1024 and step % 2 == 0:
            data = buffer.sample()
            _, train_metric = agent.train(data)
            if step % 100 == 0:
                logger.log(train_metric, step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    main(args)
