"""
TODO: Write img_o and obs_o more succinctly
"""

import math
from functools import partial

import jax.numpy as jnp
from chex import Array, ArrayTree
from distrax import Independent
from flax import nnx
from jax.lax import stop_gradient
from jax.tree_util import tree_map

from dreamerv3_flax.distribution import Dist, OneHotCategorical
from dreamerv3_flax.flax_util import Linear
from dreamerv3_flax.jax_util import where


class RSSM(nnx.Module):
    def __init__(
        self,
        in_size: int,
        num_actions: int,
        *,
        hid_size: int = 1024,
        deter_size: int = 4096,
        stoch_size: int = 32,
        num_classes: int = 32,
        uniform_mix: float = 0.01,
        act_type: str = "silu",
        norm_type: str = "layer",
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs
    ):
        # Arguments
        self.deter_size = deter_size
        self.stoch_shape = (stoch_size, num_classes)
        self.uniform_mix = uniform_mix
        self.rngs = rngs

        # State
        self.deter = nnx.Param(jnp.zeros((deter_size,)))

        # GRU
        self.gru_x_linear = Linear(
            hid_size,
            3 * deter_size,
            act_type="none",
            norm_type=norm_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )
        self.gru_h_linear = Linear(
            deter_size,
            3 * deter_size,
            act_type="none",
            norm_type=norm_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )

        # Imagination
        self.img_i_dense = Linear(
            stoch_size + num_actions,
            hid_size,
            act_type=act_type,
            norm_type=norm_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )
        self.img_o_dense = Linear(
            deter_size,
            hid_size,
            act_type=act_type,
            norm_type=norm_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )
        self.img_o_linear = Linear(
            hid_size,
            math.prod(self.stoch_shape),
            act_type="none",
            norm_type="none",
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )

        # Observation
        self.obs_o_dense = Linear(
            deter_size + in_size,
            hid_size,
            act_type=act_type,
            norm_type=norm_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )
        self.obs_o_linear = Linear(
            hid_size,
            math.prod(self.stoch_shape),
            act_type="none",
            norm_type="none",
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )

    def latent_size(self) -> int:
        return self.deter_size + math.prod(self.stoch_shape)

    def initial_state(self, batch_size: int) -> ArrayTree:
        deter = self.deter.repeat(batch_size, axis=0)
        deter = jnp.tanh(deter)
        deter = deter.astype(jnp.bfloat16)
        x = self.img_o_dense(deter)
        logit = self.img_o_linear(x)
        logit = logit.reshape(*logit.shape[:-1], *self.stoch_shape)
        dist = self.get_dist(logit)
        stoch = dist.mode()
        stoch = stoch.astype(jnp.bfloat16)
        state = {"deter": deter, "logit": logit, "stoch": stoch}
        return state

    def get_dist(self, logit: Array) -> Dist:
        logit = logit.astype(jnp.float32)
        dist = OneHotCategorical(logit, uniform_mix=self.uniform_mix)
        dist = Independent(dist, reinterpreted_batch_ndims=1)
        return dist

    def get_latent(self, state: ArrayTree) -> Array:
        deter = state["deter"]
        stoch = state["stoch"]
        stoch = stoch.reshape(*stoch.shape[:-2], -1)
        latent = jnp.concatenate([deter, stoch], axis=-1)
        return latent

    def gru(self, deter: Array, x: Array) -> Array:
        x = self.gru_x_linear(x)
        h = self.gru_h_linear(deter)
        reset_x, update_x, cand_x = jnp.split(x, 3, axis=-1)
        reset_h, update_h, cand_h = jnp.split(h, 3, axis=-1)
        reset = nnx.sigmoid(reset_x + reset_h)
        update = nnx.sigmoid(update_x + update_h)
        cand = nnx.tanh(cand_x + reset * cand_h)
        deter = (1 - update) * cand + update * deter
        return deter

    def img_step(self, state: ArrayTree, action: Array) -> ArrayTree:
        # GRU
        stoch = state["stoch"]
        stoch = jnp.reshape(stoch, (*stoch.shape[:-2], -1))
        action = action.astype(jnp.bfloat16)
        x = jnp.concatenate([stoch, action], axis=-1)
        x = self.img_i_dense(x)
        deter = state["deter"]
        deter = self.gru(deter, x)

        # Prior
        x = self.img_o_dense(deter)
        logit = self.img_o_linear(x)
        logit = jnp.reshape(logit, (*logit.shape[:-1], *self.stoch_shape))
        dist = self.get_dist(logit)
        seed = self.rngs.prior()
        stoch = dist.sample(seed=seed)
        stoch = stoch.astype(jnp.bfloat16)
        prior = {"deter": deter, "logit": logit, "stoch": stoch}

        return prior

    def obs_step(
        self,
        state: ArrayTree,
        action: Array,
        encoded: Array,
        first: Array,
    ) -> ArrayTree:
        # Masking
        action = action.astype(jnp.bfloat16)
        first = first.astype(jnp.bfloat16)
        condition = 1.0 - first
        initial_state = self.initial_state(first.shape[0])
        state = tree_map(partial(where, condition), state, initial_state)
        action = where(condition, action)

        # Imagination
        prior = self.img_step(state, action)

        # Post
        deter = prior["deter"]
        x = jnp.concatenate([deter, encoded], axis=-1)
        x = self.obs_o_dense(x)
        logit = self.obs_o_linear(x)
        logit = jnp.reshape(logit, (*logit.shape[:-1], *self.stoch_shape))
        dist = self.get_dist(logit)
        seed = self.rngs.post()
        stoch = dist.sample(seed=seed)
        stoch = stoch.astype(jnp.bfloat16)
        post = {"deter": deter, "logit": logit, "stoch": stoch}

        return post, prior

    def dyn_loss(self, post: ArrayTree, prior: ArrayTree) -> ArrayTree:
        """Calculates the dynamic loss and posterior entropy."""
        # Get the posterior and prior logits.
        post_logit = post["logit"]
        prior_logit = prior["logit"]

        # Apply the stop gradient to the posterior logit.
        post_logit = stop_gradient(post_logit)

        # Get the posterior and prior distributions.
        post_dist = self.get_dist(post_logit)
        prior_dist = self.get_dist(prior_logit)

        # Calculate the dynamic loss.
        dyn_loss = post_dist.kl_divergence(prior_dist)
        dyn_loss = jnp.maximum(dyn_loss, 1.0)

        # Calculate the posterior entropy.
        post_ent = post_dist.entropy()

        return dyn_loss, post_ent

    def rep_loss(self, post: ArrayTree, prior: ArrayTree) -> ArrayTree:
        """Calculates the representation loss and prior entropy."""
        # Get the posterior and prior logits.
        post_logit = post["logit"]
        prior_logit = prior["logit"]

        # Apply the stop gradient to the prior logit.
        prior_logit = stop_gradient(prior_logit)

        # Get the posterior and prior distributions.
        post_dist = self.get_dist(post_logit)
        prior_dist = self.get_dist(prior_logit)

        # Calculate the representation loss.
        rep_loss = post_dist.kl_divergence(prior_dist)
        rep_loss = jnp.maximum(rep_loss, 1.0)

        # Calculate the prior entropy.
        prior_ent = prior_dist.entropy()

        return rep_loss, prior_ent
