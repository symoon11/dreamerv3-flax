import math
from typing import Sequence

from chex import Array
import jax.numpy as jnp
import flax.linen as nn

from dreamerv3_flax.flax_util import Conv


class CNNEncoder(nn.Module):
    """CNN encoder module."""

    in_shape: Sequence[int]
    chan: int = 96
    min_res: int = 4
    act_type: str = "silu"
    norm_type: str = "layer"

    def setup(self):
        """Initializes an encoder."""
        # Convolutional layers
        num_layers = int(math.log2(self.in_shape[0] // self.min_res))
        out_chans = [2**i * self.chan for i in range(num_layers)]
        self.layers = [
            Conv(
                out_chan,
                kernel_size=(4, 4),
                strides=(2, 2),
                act_type=self.act_type,
                norm_type=self.norm_type,
            )
            for out_chan in out_chans
        ]

    def __call__(self, x: Array) -> Array:
        """Runs the forward pass of the encoder."""
        # Transform the input.
        x = x - 0.5

        # Apply the convolutional layers.
        for layer in self.layers:
            x = layer(x)
        x = jnp.reshape(x, (*x.shape[:-3], -1))

        return x
