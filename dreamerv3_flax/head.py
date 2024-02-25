import math
from typing import Sequence

from chex import Array
from distrax import Bernoulli, Independent
import flax.linen as nn
import jax.numpy as jnp

from dreamerv3_flax.distribution import Discrete, Dist, MSE, OneHotCategorical
from dreamerv3_flax.flax_util import Dense
from dreamerv3_flax.mlp import MLP


class MLPHead(nn.Module):
    """
    MLP head module.
    """

    out_shape: Sequence[int]
    hid_size: int = 1024
    num_layers: int = 5
    act_type: str = "silu"
    norm_type: str = "layer"
    scale: float = 1.0
    dist_type: str = "bernoulli"
    uniform_mix: float = 0.01
    num_bins: int = 255
    low: float = -20.0
    high: float = 20.0
    trans_type: str = "none"

    def setup(self):
        """Initializes a head."""
        # MLP
        self.mlp = MLP(
            hid_size=self.hid_size,
            num_layers=self.num_layers,
            act_type=self.act_type,
            norm_type=self.norm_type,
        )

        # Linear layer
        self.logit_shape = self.out_shape
        if self.dist_type == "discrete":
            self.logit_shape = (*self.logit_shape, self.num_bins)
        self.linear = Dense(
            math.prod(self.logit_shape),
            act_type="none",
            norm_type="none",
            scale=self.scale,
        )

        # Distribution
        self.event_ndims = len(self.out_shape)
        if self.dist_type == "categorical":
            self.event_ndims -= 1

    def get_dist(self, x: Array) -> Dist:
        """Returns the distribution given the input."""
        # Cast the input to float32.
        x = jnp.astype(x, jnp.float32)

        # Get the distribution.
        if self.dist_type == "bernoulli":
            dist = Bernoulli(x)
        elif self.dist_type == "categorical":
            dist = OneHotCategorical(x, uniform_mix=self.uniform_mix)
        elif self.dist_type == "discrete":
            dist = Discrete(x, low=self.low, high=self.high, trans_type=self.trans_type)
        elif self.dist_type == "mse":
            dist = MSE(x, trans_type=self.trans_type)
        else:
            raise NotImplementedError(self.dist_type)
        if self.event_ndims > 0:
            dist = Independent(dist, reinterpreted_batch_ndims=self.event_ndims)

        return dist

    def __call__(self, x: Array) -> Dist:
        """Runs the forward pass of the head."""
        # Apply the MLP.
        x = self.mlp(x)

        # Apply the linear layer.
        x = self.linear(x)
        x = jnp.reshape(x, (*x.shape[:-1], *self.logit_shape))

        # Get the distribution.
        dist = self.get_dist(x)

        return dist
