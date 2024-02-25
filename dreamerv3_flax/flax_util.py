from typing import Sequence

from chex import Array
import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax.numpy as jnp


class BaseLayer(nn.Module):
    """Base layer module."""

    out_size: int
    act_type: str = "silu"
    norm_type: str = "layer"
    scale: float = 1.0

    def setup(self):
        """Initializes a layer."""
        # Normalization
        if self.norm_type == "none":
            self.norm = None
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(dtype=jnp.float16)
        else:
            raise NotImplementedError(self.norm_type)

        # Layer
        use_bias = self.norm is None
        kernel_init = variance_scaling(
            self.scale,
            mode="fan_avg",
            distribution="truncated_normal",
        )
        self.layer = None
        self.layer_kwargs = {
            "use_bias": use_bias,
            "kernel_init": kernel_init,
            "dtype": jnp.float16,
        }

        # Activation
        if self.act_type == "none":
            self.act = None
        elif self.act_type == "silu":
            self.act = nn.silu
        elif self.act_type == "relu":
            self.act = nn.relu
        else:
            raise NotImplementedError(self.act_type)

    def __call__(self, x: Array) -> Array:
        """Runs the forward pass of the layer."""
        # Apply the layer.
        x = self.layer(x)

        # Apply the normalization.
        if self.norm:
            x = self.norm(x)

        # Apply the activation.
        if self.act:
            x = self.act(x)

        return x


class Dense(BaseLayer):
    """Dense layer module."""

    def setup(self):
        """Initializes a dense layer."""
        super().setup()

        # Layer
        self.layer = nn.Dense(self.out_size, **self.layer_kwargs)


class Conv(BaseLayer):
    """Convolutional layer module."""

    kernel_size: Sequence[int] = (4, 4)
    strides: Sequence[int] = (2, 2)

    def setup(self):
        """Initializes a convolutional layer."""
        super().setup()

        # Layer
        self.layer = nn.Conv(
            self.out_size,
            self.kernel_size,
            strides=self.strides,
            **self.layer_kwargs,
        )


class ConvTranspose(BaseLayer):
    """Transposed convolutional layer module."""

    kernel_size: Sequence[int] = (4, 4)
    strides: Sequence[int] = (2, 2)

    def setup(self):
        """Initializes a transposed convolutional layer."""
        super().setup()

        # Layer
        self.layer = nn.ConvTranspose(
            self.out_size,
            self.kernel_size,
            strides=self.strides,
            **self.layer_kwargs,
        )
