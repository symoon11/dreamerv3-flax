from chex import Array, ArrayTree
import numpy as np

import crafter
from gym.vector import VectorEnvWrapper


TASKS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]


class CrafterEnv(crafter.Env):
    """Crafter environment."""

    _done: bool = True
    _episode_return: float = 0
    _episode_length: int = 0

    def reset(self) -> ArrayTree:
        raise NotImplementedError

    def step(self, action: int) -> ArrayTree:
        """Steps the environment."""
        if self._done:
            # Reset the environment.
            obs = super().reset()
            reward = 0.0
            done = False
            info = {
                "inventory": self._player.inventory.copy(),
                "achievements": self._player.achievements.copy(),
                "discount": 1.0,
                "semantic": self._sem_view(),
                "player_pos": self._player.pos,
                "reward": reward,
                "first": True,
            }

            # Update the statistics.
            self._done = done
            self._episode_return = 0
            self._episode_length = 0

        else:
            # Step the environment.
            obs, reward, done, info = super().step(action)
            info = {**info, "first": False}

            # Update the statistics.
            self._done = done
            self._episode_return += reward
            self._episode_length += 1

            # Return the episode return and length.
            if done:
                info["episode_return"] = self._episode_return
                info["episode_length"] = self._episode_length

        return obs, reward, done, info


class VecCrafterEnv(VectorEnvWrapper):
    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def single_obseravtion_space(self):
        return self.env.single_observation_space

    @property
    def single_action_space(self):
        return self.env.single_action_space

    @staticmethod
    def transform_dones(dones: Array) -> Array:
        """Transforms dones."""
        dones = dones.astype(np.float32)
        return dones

    @staticmethod
    def get_firsts(infos: ArrayTree) -> Array:
        """Returns firsts."""
        firsts = [info["first"] for info in infos]
        firsts = np.array(firsts, dtype=np.float32)
        return firsts

    def step_wait(self) -> ArrayTree:
        """Steps the environment."""
        obs, rewards, dones, infos = self.env.step_wait()
        dones = self.transform_dones(dones)
        firsts = self.get_firsts(infos)
        return obs, rewards, dones, firsts, infos
