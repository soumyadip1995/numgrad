import numpy as np
import torch

# from evcu import Variable, dot, relu, sumel, backward_graph
from engine import Tensor, dot, relu, backward_graph



# l1 = Variable(np.arange(-4,4).reshape(2,4))
# l2 = Variable(np.arange(-2,2).reshape(4,1))
# n1 = dot(l1,l2)
# n2 = relu(n1)
# n3 = sumel(n2)
# backward_graph(n2)
# print(l1.grad, l2.grad)


# x = np.random.randn(3,3).astype(np.float32)
# W = np.random.randn(3,2).astype(np.float32)
# m = np.random.randn(1,3).astype(np.float32)
# y = Tensor(x)

# z = Tensor(x)
# print(x)
# a = Tensor(W)
# b = Tensor(m)
# out = x.dot(W)


## Original test
# x = Variable(np.random.randn(3,3).astype(np.float32))
# y = Variable(np.random.randn(3,3).astype(np.float32))

# n = dot(x, y)
# p = relu(n)
# w = sumel(p)
# backward_graph(p)
# print(x.grad)
# print(y.grad)

# Original Test

# l1 = Variable(np.arange(-4,4).reshape(2,4))
# l2 = Variable(np.arange(-2,2).reshape(4,1))
# n1 = dot(l1,l2)
# n2 = relu(n1)
# n3 = sumel(n2)
# backward_graph(n2)
# print(l1.grad, l2.grad)

# My test

l1 = np.random.randn(3,3).astype(np.float32)
l2 = np.random.randn(3,3).astype(np.float32)

def engine():

	x = Tensor(l1)
	W = Tensor(l2)
	n1 = dot(x, W)
	n2 = relu(n1)
	backward_graph(n2)
	return x.grad, W.grad

def pytorch():

    x = torch.tensor(l1, requires_grad=True)
    W = torch.tensor(l2, requires_grad=True)
    output = x.matmul(W)
    output1 = output.relu()
    output1.sum().backward()
    return output.detach().numpy() , x.grad, W.grad


for x,y in zip(engine(), pytorch()):
  print(x,y)
  


# My test

# l1 = Tensor(np.arange(-4,4).reshape(2,4))
# l2 = Tensor(np.arange(-2,2).reshape(4,1))
# n1 = dot(l1,l2)
# n2 = relu(n1)
# print(l1.grad, n2)