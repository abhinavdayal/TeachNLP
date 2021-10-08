import numpy as np
from .computation_graph import ComputationGraph
"""
Creating new data structure.... It should support gradients passing and computation graph
"""
class Vensor:
    def __init__(self, data, requires_grad = False, is_leaf = True):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf

    def __getitem__(self, key):
        return self.data[key]

    
    def __convert(self, data):
        return Vensor(data, requires_grad=self.requires_grad, is_leaf=False)

    def __add__(self, other):
        
        if(type(other) != type(self)): # check : is the type of other compatible
            other = self.__convert(other)
        res  = Vensor(self.data+other.data)
        ComputationGraph.add(res, self, np.ones_like(other.data), other, np.ones_like(self.data))
        return res

    def __radd__(self, other):
        if(type(other) != type(self)):  # check : is the type of other compatible
            other = self.__convert(other)
        res = Vensor(self.data+other.data)
        ComputationGraph.add(res, self, np.ones_like(
            other.data), other, np.ones_like(self.data))
        return res

    def __mul__(self, other):
        if(type(other) != type(self)):  # check : is the type of other compatible
            other = self.__convert(other)
        res = Vensor(self.data*other.data)
        ComputationGraph.add(res, self, other.data, other, self.data)
        return res

    def __repr__(self):
        return f"Vensor( {self.data} )"

    def backward(self):
        ComputationGraph.backward(np.array([1]),self)
    
    def sum(self):
        res = Vensor(np.sum(self.data))
        ComputationGraph.add(res, self, np.ones_like(self.data))
        return res 

