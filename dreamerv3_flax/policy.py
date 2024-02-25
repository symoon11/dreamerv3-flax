from typing import Dict

from chex import Array, ArrayTree
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
from jax.lax import stop_gradient
import jax.numpy as jnp
from jax.tree_util import tree_map

from dreamerv3_flax.head import MLPHead
from dreamerv3_flax.normalizer import Normalizer


class Policy(nn.Module):
    """Policy module."""

    num_actions: int
    gamma: float = 0.997
    gae_lambda: float = 0.95
    action_ent_coef: float = 3e-4
    value_reg_coef: float = 1.0
    update_freq: int = 1
    decay: float = 0.98
    action_head_kwargs: Dict = FrozenDict(
        hid_size=1024,
        num_layers=5,
        act_type="silu",
        norm_type="layer",
        scale=1.0,
        dist_type="categorical",
        uniform_mix=0.01,
    )
    value_head_kwargs: Dict = FrozenDict(
        hid_size=1024,
        num_layers=5,
        act_type="silu",
        norm_type="layer",
        scale=0.0,
        dist_type="discrete",
        low=-20.0,
        high=20.0,
        trans_type="symlog",
    )
    normalizer_kwargs: Dict = FrozenDict(
        decay=0.99,
        max_scale=1.0,
        q_low=5.0,
        q_high=95.0,
    )

    def setup(self):
        """Initializes a policy."""
        # Action head
        self.action_head = MLPHead((self.num_actions,), **self.action_head_kwargs)

        # Value head
        self.value_head = MLPHead((), **self.value_head_kwargs)

        # Slow value head
        self.slow_value_head = MLPHead((), **self.value_head_kwargs)

        # Normalizer
        self.normalizer = Normalizer(**self.normalizer_kwargs)

        # Counter
        self.counter = self.variable("stats", "counter", jnp.zeros, (), jnp.float32)

    def act(self, latent: Array) -> Array:
        """Samples an action."""
        action_dist = self.action_head(latent)
        seed = self.make_rng("action")
        action = action_dist.sample(seed=seed)
        return action

    def get_value(self, latent: Array) -> Array:
        """Calculates the value."""
        value_dist = self.value_head(latent)
        value = value_dist.mean()
        return value

    def get_slow_value(self, latent: Array) -> Array:
        """Calculates the slow value."""
        slow_value_dist = self.slow_value_head(latent)
        slow_value = slow_value_dist.mean()
        return slow_value

    def get_vtarg(self, latent: Array, reward: Array, cont: Array) -> ArrayTree:
        """Calculates the value target."""
        # Calculate the value.
        value = self.get_value(latent)

        # Calculate the value target.
        vtarg = [value[-1]]
        discount = self.gamma * cont[1:]
        interm = reward + discount * value[1:] * (1 - self.gae_lambda)
        for t in reversed(range(discount.shape[0])):
            vtarg.append(interm[t] + discount[t] * self.gae_lambda * vtarg[-1])
        vtarg = jnp.stack(list(reversed(vtarg)), axis=0)

        # Drop the last timestep.
        vtarg = vtarg[:-1]
        value = value[:-1]

        return vtarg, value

    def get_adv(self, vtarg: Array, value: Array) -> Array:
        """Calculates the advantage."""
        # Normalize the return and value.
        offset, inv_scale = self.normalizer(vtarg)
        vtarg = (vtarg - offset) / inv_scale
        value = (value - offset) / inv_scale

        # Calculate the advantage.
        adv = vtarg - value

        return adv

    def get_weight(self, cont: Array) -> Array:
        """Calculates the weight."""
        weight = jnp.cumprod(self.gamma * cont, axis=0) / self.gamma
        return weight

    def action_loss(self, latent: Array, action: Array, adv: Array) -> ArrayTree:
        """Calculates the action loss and entropy."""
        # Apply the stop gradient to the advantage.
        adv = stop_gradient(adv)

        # Calculate the action loss and entropy.
        action_dist = self.action_head(latent)
        action_loss = -adv * action_dist.log_prob(action)
        action_ent = action_dist.entropy()

        return action_loss, action_ent

    def value_loss(self, latent: Array, vtarg: Array, slow_value: Array) -> ArrayTree:
        """Calculates the value loss and regularizer."""
        # Apply the stop gradient to the value target and slow value.
        vtarg = stop_gradient(vtarg)
        slow_value = stop_gradient(slow_value)

        # Calculate the value loss and regularizer.
        value_dist = self.value_head(latent)
        value_loss = -value_dist.log_prob(vtarg)
        value_reg = -value_dist.log_prob(slow_value)

        return value_loss, value_reg

    def actor_loss(
        self,
        latent: Array,
        action: Array,
        cont: Array,
        adv: Array,
    ) -> ArrayTree:
        """Calculates the actor loss."""
        # Drop the last timestep.
        latent = latent[:-1]
        action = action[:-1]
        cont = cont[:-1]

        # Calculate the action loss and entropy.
        action_loss, action_ent = self.action_loss(latent, action, adv)

        # Calculate the actor loss.
        actor_loss = action_loss - self.action_ent_coef * action_ent

        # Calculate the weighted mean.
        weight = self.get_weight(cont)
        actor_loss = jnp.mean(weight * actor_loss)

        # Define the actor metric.
        actor_metric = {
            "action_loss": jnp.mean(action_loss),
            "action_loss_hist": jnp.ravel(action_loss),
            "action_ent": jnp.mean(action_ent),
            "action_ent_hist": jnp.ravel(action_ent),
        }

        return actor_loss, actor_metric

    def critic_loss(
        self,
        latent: Array,
        cont: Array,
        vtarg: Array,
        slow_value: ArrayTree,
    ) -> ArrayTree:
        """Calculates the critic loss."""
        # Drop the last timestep.
        latent = latent[:-1]
        cont = cont[:-1]
        slow_value = slow_value[:-1]

        # Calculate the value loss and regularizer.
        value_loss, value_reg = self.value_loss(latent, vtarg, slow_value)

        # Calculate the critic loss.
        critic_loss = value_loss + self.value_reg_coef * value_reg

        # Calculate the weighted mean.
        weight = self.get_weight(cont)
        critic_loss = jnp.mean(weight * critic_loss)

        # Define the critic metric.
        critic_metric = {
            "value_loss": jnp.mean(value_loss),
            "value_loss_hist": jnp.ravel(value_loss),
            "value_reg": jnp.mean(value_reg),
            "value_reg_hist": jnp.ravel(value_reg),
        }

        return critic_loss, critic_metric

    def policy_loss(
        self,
        latent: Array,
        action: Array,
        reward: Array,
        cont: Array,
    ) -> ArrayTree:
        """Calculates the policy loss."""
        # Calculate the return and value.
        vtarg, value = self.get_vtarg(latent, reward, cont)

        # Calculate the advantage.
        adv = self.get_adv(vtarg, value)

        # Calculate the slow value.
        slow_value = self.get_slow_value(latent)

        # Calculate the actor loss.
        actor_loss, actor_metric = self.actor_loss(latent, action, cont, adv)

        # Calculate the critic loss.
        critic_loss, critic_metric = self.critic_loss(latent, cont, vtarg, slow_value)

        # Calculate the policy loss.
        policy_loss = actor_loss + critic_loss

        # Define the policy metric.
        policy_metric = {
            **actor_metric,
            **critic_metric,
            "vtarg": jnp.mean(vtarg),
            "vtarg_hist": jnp.ravel(vtarg),
            "value": jnp.mean(value),
            "value_hist": jnp.ravel(value),
            "adv": jnp.mean(adv),
            "adv_hist": jnp.ravel(adv),
            "slow_value": jnp.mean(slow_value),
            "slow_value_hist": jnp.ravel(slow_value),
            "reward": jnp.mean(reward),
            "reward_hist": jnp.ravel(reward),
            "cont": jnp.mean(cont),
            "cont_hist": jnp.ravel(cont),
        }

        return policy_loss, policy_metric

    def update_policy(self):
        """Updates the policy."""
        # Get the counter.
        counter = self.counter.value

        # Calculate the mixing ratio.
        init = jnp.astype(counter == 0, jnp.float32)
        update = jnp.astype(counter % self.update_freq == 0, jnp.float32)
        mix = jnp.clip(init * 1.0 + update * (1 - self.decay), 0.0, 1.0)

        # Update the slow value head.
        def update(x: Array, y: Array) -> Array:
            return (1 - mix) * x + mix * y

        params = self.value_head.variables.get("params")
        slow_params = self.slow_value_head.variables.get("params")
        for key in slow_params.keys():
            param = params.get(key)
            slow_param = slow_params.get(key)
            slow_param = tree_map(update, slow_param, param)
            self.slow_value_head.put_variable("params", key, slow_param)

        # Update the counter.
        self.counter.value = counter + 1
