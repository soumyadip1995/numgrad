### numgrad

If torch is based on the Torch Tensor, then numgrad is based on the numpy array. A Tensor class wrapping the numpy array. If [karpathy/micrograd](https://github.com/karpathy/micrograd) provides support for scalar values and its gradients, numgrad provides support for both scalar values and matrices.

#### One Example

``` 

x = Tensor(np.arange(-4, 4).reshape(2, 4))
y = Tensor(np.arange(-2, 2).reshape(4, 1))
n = dot(x, y)
n1 = relu(n)
backward_graph(n1)
print(x.grad, y.grad)

``` 

#### Scalar values

Some scalar computation. Not final

``` 

a = Tensor(-8.0)
b = Tensor(2.0)
c = add(a, b)
d = relu(c)
d.backward()
print(a.grad, b.grad)

