import math
from typing import Sequence

from chex import Array
from distrax import Independent
import flax.linen as nn
import jax.numpy as jnp

from dreamerv3_flax.distribution import Dist, MSE
from dreamerv3_flax.flax_util import ConvTranspose, Dense


class CNNDecoder(nn.Module):
    """CNN decoder module."""

    out_shape: Sequence[int]
    chan: int = 96
    min_res: int = 4
    act_type: str = "silu"
    norm_type: str = "layer"

    def setup(self):
        """Initializes a decoder."""
        # Linear layer
        num_layers = int(math.log2(self.out_shape[0] // self.min_res))
        in_chan = 2 ** (num_layers - 1) * self.chan
        self.in_shape = (self.min_res, self.min_res, in_chan)
        self.linear = Dense(
            math.prod(self.in_shape),
            act_type="none",
            norm_type="none",
        )

        # Convolutional layers
        out_chans = [2 ** (i - 1) * self.chan for i in reversed(range(num_layers))]
        act_types = [self.act_type for _ in range(num_layers)]
        norm_types = [self.norm_type for _ in range(num_layers)]
        out_chans[-1] = self.out_shape[-1]
        act_types[-1] = "none"
        norm_types[-1] = "none"
        self.layers = [
            ConvTranspose(
                out_chan,
                kernel_size=(4, 4),
                strides=(2, 2),
                act_type=act_type,
                norm_type=norm_type,
            )
            for out_chan, act_type, norm_type in zip(out_chans, act_types, norm_types)
        ]

    def get_dist(self, loc: Array) -> Dist:
        """Returns the distribution."""
        # Cast the location to float32.
        loc = jnp.astype(loc, jnp.float32)

        # Get the distribution.
        dist = MSE(loc)
        dist = Independent(dist, reinterpreted_batch_ndims=len(self.out_shape))

        return dist

    def __call__(self, x: Array) -> Dist:
        """Runs the forward pass of the decoder."""
        # Apply the linear layer.
        x = self.linear(x)
        x = jnp.reshape(x, (*x.shape[:-1], *self.in_shape))

        # Apply the convolutional layers.
        for layer in self.layers:
            x = layer(x)

        # Calculate the location.
        loc = x + 0.5

        # Get the distribution.
        dist = self.get_dist(loc)

        return dist
