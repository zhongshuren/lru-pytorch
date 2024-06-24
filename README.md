# lru-pytorch

Pytorch implementation of LRU(Linear Recurrent Units) with parallelized scan.

Verified with copy task from https://github.com/NicolasZucchet/minimal-LRU, which is originally implemented in JAX.

Original LRU paper: <a href='https://arxiv.org/abs/2303.06349'>Resurrecting Recurrent Neural Networks for Long Sequences</a>

## Example

```python
import torch
from lru import LRU
lru = LRU(d_model=32, d_hidden=16)
x = torch.rand(48, 50, 32)
# Note that the shape of input tensor is (L, B, D).
y = lru(x)
print(y.shape)
```
The output should be:
```text
torch.Size([48, 50, 32])
```

## Run Copy Task

```shell
python main.py
```


## Reference Code
https://github.com/NicolasZucchet/minimal-LRU
https://github.com/Gothos/LRU-pytorch
https://github.com/yueqirex/LRURec