from functools import partial
import math

from chex import Array, ArrayTree
from distrax import Independent
import flax.linen as nn
from flax.linen.initializers import zeros_init
from jax.lax import stop_gradient
import jax.numpy as jnp
from jax.tree_util import tree_map

from dreamerv3_flax.distribution import Dist, OneHotCategorical
from dreamerv3_flax.flax_util import Dense
from dreamerv3_flax.jax_util import where


class RSSM(nn.Module):
    """RSSM module."""

    hid_size: int = 1024
    deter_size: int = 4096
    stoch_size: int = 32
    num_classes: int = 32
    uniform_mix: float = 0.01
    act_type: str = "silu"
    norm_type: str = "layer"

    def setup(self):
        """Initializes a RSSM."""
        # State
        self.deter = self.param("deter", zeros_init(), (self.deter_size,))
        self.logit_shape = (self.stoch_size, self.num_classes)

        # GRU
        self.gru_x_linear = Dense(
            3 * self.deter_size,
            act_type="none",
            norm_type=self.norm_type,
        )
        self.gru_h_linear = Dense(
            3 * self.deter_size,
            act_type="none",
            norm_type=self.norm_type,
        )

        # Imagination
        self.img_i_dense = Dense(
            self.hid_size,
            act_type=self.act_type,
            norm_type=self.norm_type,
        )
        self.img_o_dense = Dense(
            self.hid_size,
            act_type=self.act_type,
            norm_type=self.norm_type,
        )
        self.img_o_linear = Dense(
            math.prod(self.logit_shape),
            act_type="none",
            norm_type="none",
        )

        # Observation
        self.obs_o_dense = Dense(
            self.hid_size,
            act_type=self.act_type,
            norm_type=self.norm_type,
        )
        self.obs_o_linear = Dense(
            math.prod(self.logit_shape),
            act_type="none",
            norm_type="none",
        )

    def latent_size(self) -> int:
        """Returns the latent size."""
        return self.deter_size + self.stoch_size * self.num_classes

    def initial_state(self, batch_size: int) -> ArrayTree:
        """Returns the initial RSSM state."""
        # Get the deterministic representations.
        deter = self.deter[None].repeat(batch_size, axis=0)
        deter = jnp.tanh(deter)

        # Cast the deterministic representation to float16.
        deter = jnp.astype(deter, jnp.float16)

        # Calculate the logit.
        x = self.img_o_dense(deter)
        logit = self.img_o_linear(x)
        logit = logit.reshape(*logit.shape[:-1], *self.logit_shape)

        # Sample a stochastic representation.
        dist = self.get_dist(logit)
        stoch = dist.mode()

        # Cast the stochastic representation to float16.
        stoch = jnp.astype(stoch, jnp.float16)

        # Define the initial RSSM state.
        state = {"deter": deter, "logit": logit, "stoch": stoch}

        return state

    def get_dist(self, logit: Array) -> Dist:
        """Returns the distribution."""
        # Cast the logit to float32.
        logit = jnp.astype(logit, jnp.float32)

        # Get the distribution.
        dist = OneHotCategorical(logit, uniform_mix=self.uniform_mix)
        dist = Independent(dist, reinterpreted_batch_ndims=1)

        return dist

    def get_latent(self, state: ArrayTree) -> Array:
        """Calculates the latent representation."""
        # Get the deterministic and stochastic representations.
        deter = state["deter"]
        stoch = state["stoch"]

        # Concatenate the deterministic and stochastic representations.
        stoch = jnp.reshape(stoch, (*stoch.shape[:-2], -1))
        latent = jnp.concatenate([deter, stoch], axis=-1)

        return latent

    def gru(self, deter: Array, x: Array) -> Array:
        """Runs the forward pass of the GRU."""
        # Apply the linear layers.
        x = self.gru_x_linear(x)
        h = self.gru_h_linear(deter)

        # Calculate the reset, update, and candidate.
        reset_x, update_x, cand_x = jnp.split(x, 3, axis=-1)
        reset_h, update_h, cand_h = jnp.split(h, 3, axis=-1)
        reset = nn.sigmoid(reset_x + reset_h)
        update = nn.sigmoid(update_x + update_h)
        cand = nn.tanh(cand_x + reset * cand_h)

        # Update the deterministic representation.
        deter = (1 - update) * cand + update * deter

        return deter

    def img_step(self, state: ArrayTree, action: Array) -> ArrayTree:
        """Runs an imagination step."""
        # Get the deterministic and stochastic representations.
        deter = state["deter"]
        stoch = state["stoch"]

        # Cast the action to float16.
        action = jnp.astype(action, jnp.float16)

        # Concatenate the stochastic representation and action.
        stoch = jnp.reshape(stoch, (*stoch.shape[:-2], -1))
        x = jnp.concatenate([stoch, action], axis=-1)

        # Update the deterministic representation.
        x = self.img_i_dense(x)
        deter = self.gru(deter, x)

        # Calculate the logit.
        x = self.img_o_dense(deter)
        logit = self.img_o_linear(x)
        logit = jnp.reshape(logit, (*logit.shape[:-1], *self.logit_shape))

        # Sample a stochastic representation.
        dist = self.get_dist(logit)
        seed = self.make_rng("prior")
        stoch = dist.sample(seed=seed)

        # Cast the stochastic representation to float16.
        stoch = jnp.astype(stoch, jnp.float16)

        # Define the prior state.
        prior = {"deter": deter, "logit": logit, "stoch": stoch}

        return prior

    def obs_step(
        self,
        state: ArrayTree,
        action: Array,
        encoded: Array,
        first: Array,
    ) -> ArrayTree:
        """Runs an observation step."""
        # Cast the action and first to float16.
        action = jnp.astype(action, jnp.float16)
        first = jnp.astype(first, jnp.float16)

        # Mask the state and action.
        condition = 1.0 - first
        initial_state = self.initial_state(first.shape[0])
        state = tree_map(partial(where, condition), state, initial_state)
        action = where(condition, action)

        # Run an imagination step.
        prior = self.img_step(state, action)

        # Concatenate the deterministic and encoded representations.
        deter = prior["deter"]
        x = jnp.concatenate([deter, encoded], axis=-1)

        # Calculate the logit.
        x = self.obs_o_dense(x)
        logit = self.obs_o_linear(x)
        logit = jnp.reshape(logit, (*logit.shape[:-1], *self.logit_shape))

        # Sample a stochastic representation.
        dist = self.get_dist(logit)
        seed = self.make_rng("post")
        stoch = dist.sample(seed=seed)

        # Cast the stochastic representation to float16.
        stoch = jnp.astype(stoch, jnp.float16)

        # Define the posterior state.
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
