from typing import Dict, Sequence

from chex import Array, ArrayTree
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import jax.numpy as jnp

from dreamerv3_flax.decoder import CNNDecoder
from dreamerv3_flax.distribution import Dist
from dreamerv3_flax.encoder import CNNEncoder
from dreamerv3_flax.head import MLPHead
from dreamerv3_flax.rssm import RSSM


class WorldModel(nn.Module):
    """World model module."""

    obs_shape: Sequence[int]
    num_actions: int
    encoder_kwargs: Dict = FrozenDict(
        chan=96,
        min_res=4,
        act_type="silu",
        norm_type="layer",
    )
    rssm_kwargs: Dict = FrozenDict(
        hid_size=1024,
        deter_size=4096,
        stoch_size=32,
        num_classes=32,
        uniform_mix=0.01,
        act_type="silu",
        norm_type="layer",
    )
    decoder_kwargs: Dict = FrozenDict(
        chan=96,
        min_res=4,
        act_type="silu",
        norm_type="layer",
    )
    reward_head_kwargs: Dict = FrozenDict(
        hid_size=1024,
        num_layers=5,
        act_type="silu",
        norm_type="layer",
        scale=0.0,
        dist_type="discrete",
        num_bins=255,
        low=-20.0,
        high=20.0,
        trans_type="symlog",
    )
    cont_head_kwargs: Dict = FrozenDict(
        hid_size=1024,
        num_layers=5,
        act_type="silu",
        norm_type="layer",
        scale=1.0,
        dist_type="bernoulli",
    )
    loss_coef: Dict = FrozenDict(
        dyn_loss=0.5,
        rep_loss=0.1,
        obs_loss=1.0,
        reward_loss=1.0,
        cont_loss=1.0,
    )

    def setup(self):
        """Initializes a model."""
        # Encoder
        self.encoder = CNNEncoder(self.obs_shape, **self.encoder_kwargs)

        # RSSM
        self.rssm = RSSM(**self.rssm_kwargs)

        # Decoder
        self.decoder = CNNDecoder(self.obs_shape, **self.decoder_kwargs)

        # Reward head
        self.reward_head = MLPHead((), **self.reward_head_kwargs)

        # Continuation head
        self.cont_head = MLPHead((), **self.cont_head_kwargs)

    def initial_state(self, batch_size: int) -> ArrayTree:
        """Returns the initial state."""
        state = self.rssm.initial_state(batch_size)
        action = jnp.zeros((batch_size, self.num_actions), dtype=jnp.float32)
        return state, action

    def get_obs(self, latent: Array) -> Array:
        obs_dist = self.decoder(latent)
        obs = obs_dist.mode()
        return obs

    def get_reward(self, latent: Array) -> Array:
        """Calculates the reward."""
        reward_dist = self.reward_head(latent)
        reward = reward_dist.mean()
        return reward

    def get_cont(self, latent: Array) -> Array:
        """Calculates the continuation."""
        cont_dist = self.cont_head(latent)
        cont = cont_dist.mode()
        return cont

    @staticmethod
    def obs_step(
        cell: RSSM,
        state: ArrayTree,
        encoded: Array,
        action: Array,
        first: Array,
    ) -> ArrayTree:
        """Runs an observation step."""
        # Run a RSSM observation step.
        post, prior = cell.obs_step(*state, encoded, first)

        # Update the state.
        state = (post, action)

        return state, (post, prior)

    def observe(
        self,
        state: ArrayTree,
        obs: Array,
        action: Array,
        first: Array,
    ) -> ArrayTree:
        """Runs an observation."""
        # Encode the observation.
        encoded = self.encoder(obs)

        # Run an observation step.
        scan = nn.scan(
            self.obs_step,
            variable_broadcast="params",
            split_rngs={"params": False, "prior": True, "post": True},
            in_axes=1,
            out_axes=1,
        )
        state, (post, prior) = scan(self.rssm, state, encoded, action, first)

        return state, (post, prior)

    def obs_loss(self, latent: Dist, obs: Array) -> Array:
        """Calculates the observation loss."""
        obs_dist = self.decoder(latent)
        obs_loss = -obs_dist.log_prob(obs)
        return obs_loss

    def reward_loss(self, latent: Dist, reward: Array) -> Array:
        """Calculates the reward loss."""
        reward_dist = self.reward_head(latent)
        reward_loss = -reward_dist.log_prob(reward)
        return reward_loss

    def cont_loss(self, latent: Dist, cont: Array) -> Array:
        """Calculates the continuation loss."""
        cont_dist = self.cont_head(latent)
        cont_loss = -cont_dist.log_prob(cont)
        return cont_loss

    def model_loss(
        self,
        obs: Array,
        action: Array,
        reward: Array,
        cont: Array,
        first: Array,
        state: ArrayTree | None = None,
    ) -> ArrayTree:
        """Calculates the model loss."""
        # Get the initial state if none.
        if state is None:
            batch_size = first.shape[0]
            state = self.initial_state(batch_size)

        # Transform the observation.
        obs = jnp.astype(obs, jnp.float32) / 255.0

        # Run an observation.
        state, (post, prior) = self.observe(state, obs, action, first)

        # Get the latent representation.
        latent = self.rssm.get_latent(post)

        # Calculate the individual loss.
        dyn_loss, post_ent = self.rssm.dyn_loss(post, prior)
        rep_loss, prior_ent = self.rssm.rep_loss(post, prior)
        obs_loss = self.obs_loss(latent, obs)
        reward_loss = self.reward_loss(latent, reward)
        cont_loss = self.cont_loss(latent, cont)
        loss = {
            "dyn_loss": jnp.mean(dyn_loss),
            "dyn_loss_hist": jnp.ravel(dyn_loss),
            "rep_loss": jnp.mean(rep_loss),
            "rep_loss_hist": jnp.ravel(rep_loss),
            "obs_loss": jnp.mean(obs_loss),
            "obs_loss_hist": jnp.ravel(obs_loss),
            "reward_loss": jnp.mean(reward_loss),
            "reward_loss_hist": jnp.ravel(reward_loss),
            "cont_loss": jnp.mean(cont_loss),
            "cont_loss_hist": jnp.ravel(cont_loss),
        }

        # Calculate the model loss.
        model_loss = sum([v * loss[k] for k, v in self.loss_coef.items()])

        # Define the model metric.
        model_metric = {
            **loss,
            "post_ent": jnp.mean(post_ent),
            "post_ent_hist": jnp.ravel(post_ent),
            "prior_ent": jnp.mean(prior_ent),
            "prior_ent_hist": jnp.ravel(prior_ent),
        }

        return model_loss, (post, state, model_metric)
