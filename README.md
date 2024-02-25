# Flax Implementation of DreamerV3

## Modifications to the Original Implementation

1. Layer normalization: We use the default epsilon value for layer normalization from Flax.
2. GRU: We adopt the default GRU implementation from Flax.
3. Adam: We use the default epsilon value for Adam from Flax. 
4. Policy optimizer: We employ a single optimizer for the policy.
5. DynamicScale: We use the default DynamicScale for FP16 training from Optax.

## Installation
```
pip install --upgrade setuptools==65.5.0 wheel==0.38.4
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
pip install -e .
```

## Running the Training Script
```
python train.py --exp_name [exp_name]
```
