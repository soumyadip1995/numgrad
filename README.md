### numgrad

If torch is based on the Torch Tensor, then numgrad is based on the numpy array. A Tensor class wrapping the numpy array,
providing Autograd like functionality.

#### One Example

``` 

x = Tensor(np.arange(-4, 4).reshape(2, 4))
y = Tensor(np.arange(-2, 2).reshape(4, 1))
n = dot(x, y)
n1 = relu(n)
backward_graph(n1)
print(x.grad, y.grad)

``` 




