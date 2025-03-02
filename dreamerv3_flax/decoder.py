import math
from typing import Sequence

from chex import Array
import jax.numpy as jnp
from distrax import Independent
from flax import nnx

from dreamerv3_flax.distribution import MSE, Dist
from dreamerv3_flax.flax_util import ConvTranspose, Linear


class CNNDecoder(nnx.Module):
    def __init__(
        self,
        in_size: int,
        out_shape: Sequence[int],
        *,
        chan: int = 96,
        min_res: int = 4,
        act_type: str = "silu",
        norm_type: str = "layer",
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        num_layers = int(math.log2(out_shape[0] // min_res))
        in_chan = 2 ** (num_layers - 1) * chan
        self.in_shape = (min_res, min_res, in_chan)
        self.linear = Linear(
            in_size,
            math.prod(self.in_shape),
            act_type="none",
            norm_type="none",
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )
        out_chans = [2 ** (i - 1) * chan for i in reversed(range(num_layers))]
        act_types = [act_type for _ in range(num_layers)]
        norm_types = [norm_type for _ in range(num_layers)]
        out_chans[-1] = out_shape[-1]
        act_types[-1] = "none"
        norm_types[-1] = "none"
        in_chan = self.in_shape[-1]
        self.layers = []
        for out_chan in out_chans:
            layer = ConvTranspose(
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
        self.out_shape = out_shape

    def get_dist(self, loc: Array) -> Dist:
        loc = loc.astype(jnp.float32)
        dist = MSE(loc)
        dist = Independent(dist, reinterpreted_batch_ndims=len(self.out_shape))
        return dist

    def __call__(self, x: Array) -> Dist:
        x = self.linear(x)
        x = x.reshape(*x.shape[:-1], *self.in_shape)
        for layer in self.layers:
            x = layer(x)
        loc = x + 0.5
        dist = self.get_dist(loc)
        return dist
