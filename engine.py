import numpy as np
import torch
from functools import partialmethod

class Tensor:
    def __init__(self, data, leaf = True, backward_function = None):

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
        self.backward_function(dy=self.grad)
        return ("Tensor array %r with grad", self.data, self.grad)

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)


# implement some operations

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


def __top_sort(var):
    vars_seen = set()
    top_sort = []
    def top_sort_helper(vr):
        if (vr in vars_seen) or vr.leaf:
            pass
        else:
            vars_seen.add(vr)
            for pvar in vr.prev:
                top_sort_helper(pvar)
            top_sort.append(vr)
    top_sort_helper(var)
    return top_sort

def backward_graph(var):
    if not isinstance(var,Tensor):
        print("var needs to be a Tensor instance")
    tsorted = __top_sort(var)

    var.grad=np.ones(var.data.shape)
    for var in reversed(tsorted):
        var.backward()


