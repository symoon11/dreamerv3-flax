from chex import Array
import flax.linen as nn

from dreamerv3_flax.flax_util import Dense


class MLP(nn.Module):
    """MLP module."""

    hid_size: int = 1024
    num_layers: int = 5
    act_type: str = "silu"
    norm_type: str = "layer"

    def setup(self):
        """Initializes a MLP."""
        # MLP
        self.layers = [
            Dense(self.hid_size, act_type=self.act_type, norm_type=self.norm_type)
            for _ in range(self.num_layers)
        ]

    def __call__(self, x: Array) -> Array:
        """Runs the forward pass of the MLP."""
        # Apply the MLP.
        for layer in self.layers:
            x = layer(x)

        return x
