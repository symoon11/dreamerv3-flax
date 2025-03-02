import math
from typing import Sequence

import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from dreamerv3_flax.flax_util import Conv


class CNNEncoder(nnx.Module):
    def __init__(
        self,
        in_shape: Sequence[int],
        *,
        chan: int = 96,
        min_res: int = 4,
        act_type: str = "silu",
        norm_type: str = "layer",
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        num_layers = int(math.log2(in_shape[0] // min_res))
        in_chan = in_shape[-1]
        out_chans = [2**i * chan for i in range(num_layers)]
        self.layers = []
        for out_chan in out_chans:
            layer = Conv(
                in_chan,
                out_chan,
                kernel_size=(4, 4),
                strides=(2, 2),
                act_type=act_type,
                norm_type=norm_type,
                scale=scale,
                dtype=dtype,
                rngs=rngs,
            )
            self.layers.append(layer)
            in_chan = out_chan
        self.out_shape = (min_res, min_res, out_chans[-1])

    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = x - 0.5
        for layer in self.layers:
            x = layer(x)
        return x
