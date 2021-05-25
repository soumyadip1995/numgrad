## Credits to utkuevci

import numpy as np
import torch
from functools import partialmethod

class Tensor:
    def __init__(self, data, _op = '', leaf = True, backward_function = None):

        if backward_function is None and not leaf:
            print("Non-leaf nodes require backward function %r" % data)

        if np.isscalar(data):
            data = np.ones(1)*data
        if type(data) !=np.ndarray:
            print("data should be of type np.ndarray or a scalar, but received {type(data)}")
        self.leaf = leaf
        self.prev = []
        self.backward_function = backward_function
        self.data = data
        self.zero_grad()


    def backward(self):
        self.backward_function(dy = self.grad)
        

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)

    def __str__(self):
        return "Tensor %r with grad %r" % (self.data, self.grad)


    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return div

# implement some operations

def add(a,b):
    if not (isinstance(a,Tensor) and isinstance(b,Tensor)):
        print("a, b needs to be a Tensor/scalar")

    def back_func(dy):
        b.grad += dy
        a.grad += dy

    res = Tensor(a.data + b.data, leaf=False, backward_function = back_func)
    res.prev.extend([a,b])
    return res

def dot(a, b):
    if not(isinstance(a, Tensor) and isinstance(b , Tensor)):
        print("a, b needs to be a Tensor")

    def back_func(dy = 1):
        if np.isscalar(dy):
            dy = np.ones(1)*dy

        a.grad += np.dot(dy, b.data.T)
        b.grad += np.dot(a.data.T, dy)

    res = Tensor(np.dot(a.data, b.data), leaf = False, backward_function = back_func)
    res.prev.extend([a,b])

    return res

def relu(a):
    if not(isinstance(a, Tensor)):
        print("a needs to be a Tensor")

    def back_func(dy = 1):
        a.grad[a.data > 0] += dy[a.data > 0]
    res = Tensor(np.maximum(a.data, 0), leaf=False, backward_function = back_func)
    res.prev.append(a)
    return res


def mul(a,b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        print("a and b needs to be a Tensor/scalar")

    def back_func(dy):
        if np.isscalar(dy):
            dy = np.ones(1)*dy
        a.grad += np.multiply(dy,b.data)
        b.grad += np.multiply(dy,a.data)
    res = Tensor(np.multiply(a.data,b.data),'*', leaf=False, backward_function=back_func)
    res.prev.extend([a,b])
    return res


def sum(a):
    if not (isinstance(a,Tensor)):
        print('a needs to be a Tensor')
    def back_func(dy = 1):
        a.grad += np.ones(a.data.shape)*dy

    res = Tensor(np.sum(a.data), leaf=False, backward_function = back_func)
    res.prev.append(a)
    return res


def __topo_sort(var):
    vars_seen = set()
    top_sort = []
    def topo_sort_helper(vr):
        if (vr in vars_seen) or vr.leaf:
            pass
        else:
            vars_seen.add(vr)
            for pvar in vr.prev:
                topo_sort_helper(pvar)
            top_sort.append(vr)
    topo_sort_helper(var)
    return top_sort


def backward_graph(var):
    if not isinstance(var,Tensor):
        print("var needs to be a Tensor instance")
    tsorted = __topo_sort(var)

    var.grad = np.ones(var.data.shape)
    for var in reversed(tsorted):
        var.backward()


