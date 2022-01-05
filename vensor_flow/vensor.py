import numpy as np
from numpy.core.fromnumeric import shape
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
        res = self.data[key]

    
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
    
    def log(self):
        res = Vensor(np.log(self.data))
        ComputationGraph.add(res,self,1/self.data)
        return res 

    def exp(self):
        res = Vensor(np.exp(self.data))
        ComputationGraph.add(res,self,res.data)
        return res 
        
    def matmul(self, vensor):
        
        if(len(self.data.shape)!=2 or len(vensor.data.shape)!=2):
            # Raise erro
            pass 
        col1,col2 = self.data.shape[-1], vensor.data.shape[-1]
        #if(col1)



'''
v1 = Vensor([[1, 2, 3],[3,6,7]], requires_grad=True)
v2 = Vensor([[3, 5, 6],[5,8,2]], requires_grad=True)
v3 = (v1+v2)*3
v4 = v3.sum()
v4.backward()
print(v1.grad,v2.grad) 
'''
