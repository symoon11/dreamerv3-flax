from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax import nnx
from flax.linen.initializers import variance_scaling
from jax.typing import ArrayLike


class BaseLayer(nnx.Module):
    def __init__(
        self,
        out_size: int,
        act_type: str = "silu",
        norm_type: str = "layer",
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        # Normalization
        if norm_type == "none":
            self.norm = None
        elif norm_type == "layer":
            self.norm = nnx.LayerNorm(out_size, dtype=dtype, rngs=rngs)
        else:
            raise NotImplementedError(norm_type)

        # Layer
        use_bias = self.norm is None
        kernel_init = variance_scaling(
            scale, mode="fan_avg", distribution="truncated_normal"
        )
        self.layer = None
        self.layer_kwargs = {
            "use_bias": use_bias,
            "kernel_init": kernel_init,
            "dtype": dtype,
        }

        # Activation
        if act_type == "none":
            self.act = None
        elif act_type == "silu":
            self.act = nn.silu
        elif act_type == "relu":
            self.act = nn.relu
        else:
            raise NotImplementedError(act_type)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = self.layer(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class Linear(BaseLayer):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        act_type: str = "silu",
        norm_type: str = "layer",
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            out_size,
            act_type=act_type,
            norm_type=norm_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )

        # Layer
        self.layer = nnx.Linear(in_size, out_size, **self.layer_kwargs, rngs=rngs)


class Conv(BaseLayer):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        kernel_size: Sequence[int] = (4, 4),
        strides: Sequence[int] = (2, 2),
        act_type: str = "silu",
        norm_type: str = "layer",
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            out_size,
            act_type=act_type,
            norm_type=norm_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )

        # Layer
        self.layer = nnx.Conv(
            in_size,
            out_size,
            kernel_size,
            strides=strides,
            **self.layer_kwargs,
            rngs=rngs,
        )


class ConvTranspose(BaseLayer):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        kernel_size: Sequence[int] = (4, 4),
        strides: Sequence[int] = (2, 2),
        act_type: str = "silu",
        norm_type: str = "layer",
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            out_size,
            act_type=act_type,
            norm_type=norm_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )

        # Layer
        self.layer = nnx.ConvTranspose(
            in_size,
            out_size,
            kernel_size,
            strides=strides,
            **self.layer_kwargs,
            rngs=rngs,
        )
