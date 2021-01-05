### numgrad

If torch is based on the Torch Tensor, then numgrad is based on the numpy array. A Tensor class wrapping the numpy array. If [karpathy/micrograd](https://github.com/karpathy/micrograd) provides support for scalar values and its gradients, numgrad provides support for both scalar values and matrices.

#### A few Examples

A few examples have been provided below.

``` 
1)

x = Tensor(np.arange(-4, 4).reshape(2, 4))
y = Tensor(np.arange(-2, 2).reshape(4, 1))
n = dot(x, y)
n1 = relu(n)
backward_graph(n1)
print(x.grad, y.grad)

2)

x_init = np.random.randn(3,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)
x = Tensor(x_init)
y = Tensor(W_init)
c = mul(x,y)
out = relu(c)
d = sum(out)
tr = backward_graph(d)
print(out, d.data, tr)

``` 

#### Scalar values

The input can either be a scalar or a np.ndarray , some scalar value computations included. Not final.


``` 

a = Tensor(-8.0)
b = Tensor(9.0)
c = Tensor(-3.0)
outadd = add(a, b)
outm = mul(outadd, c) 
d = backward_graph(outm)
print(outm.data, outm a.grad, b.grad)

# dd/da and dd/db needs more operations. Not Final.


``` 