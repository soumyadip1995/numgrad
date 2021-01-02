import numpy as np
import torch
# from te2 import Tensor
from geo import Tensor


x_init = np.random.randn(3,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)
x = Tensor(x_init)
y = Tensor(W_init)
c = x.mul(y)
tr = c.mean().backward()
print(c, tr)


# tr = c.mean().backward()
# print(tr)


# def test_tinygrad():
#   x = Tensor(x_init)
#   print(x)
#   W = Tensor(W_init)
#   m = Tensor(m_init)
#   out = x.dot(W)
#   outr = out.relu()
#   outl = outr.logsoftmax()
#   outm = outl.mul(m)
#   outx = outm.sum()
#   outa = outm.add(m)
#   outx = outa.sum()
#   outx.backward()
#   return outx.data, x.grad, W.grad

# def test_pytorch():
#   x = torch.tensor(x_init, requires_grad=True)
#   W = torch.tensor(W_init, requires_grad=True)
#   m = torch.tensor(m_init)
#   out = x.matmul(W)
#   outr = out.relu()
#   outl = torch.nn.functional.log_softmax(outr, dim=1)
#   outm = outl.mul(m)
#   outx = outm.sum()
#   outa = outm.add(m)
#   outx = outa.sum()
#   outx.backward()
#   return outx.detach().numpy(), x.grad, W.grad



# for x,y in zip(test_tinygrad(), test_pytorch()):
#   print(x,y)
#   np.testing.assert_allclose(x, y)
