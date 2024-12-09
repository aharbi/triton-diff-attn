# triton-diff-attn
Differential Attention Kernel in Triton. Does not support masking or dropout at the moment. 

# Example Usage
The kernel can be used as follows:

```python
import torch

from layers import MultiheadDiffAttnKernel

B = 2       # Batch size
H = 16      # Number of heads
N = 512     # Sequence length
D = 32      # Head dimension

LAMBDA_SCALE = torch.tensor([0.5], dtype=torch.float16, requires_grad=True).to("cuda")
LAMBDA_INIT = 0.8
RMS_NORM = True     # RMS Normalization can be disabled

q1 = torch.randn(B, H, N, D, dtype=torch.float16, requires_grad=True, device="cuda")
q2 = torch.randn(B, H, N, D, dtype=torch.float16, requires_grad=True, device="cuda")
k1 = torch.randn(B, H, N, D, dtype=torch.float16, requires_grad=True, device="cuda")
k2 = torch.randn(B, H, N, D, dtype=torch.float16, requires_grad=True, device="cuda")
v = torch.randn(B, H, N, 2 * D, dtype=torch.float16, requires_grad=True, device="cuda")

y = MultiheadDiffAttnKernel(q1, q2, k1, k2, v, lambda_scale=LAMBDA_SCALE, rms_norm=RMS_NORM)
```
