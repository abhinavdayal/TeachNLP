class Node:
    def __init__(self, vensor1, grad1, vensor2=None, grad2=None):
        self.vensor1 = vensor1
        self.grad1 = grad1 
        self.vensor2 = vensor2
        self.grad2 = grad2

class ComputationGraph:
    nodes_dict = {}
    def __init__(self):
        pass

    @classmethod
    def add(cls, result, vensor1, grad1, vensor2 = None, grad2 = None):
        cls.nodes_dict[id(result)] = Node(vensor1, grad1, vensor2, grad2) 
    
    @classmethod
    def backward(cls, grads, vensor):
        if(id(vensor) in cls.nodes_dict):
            curr_node = cls.nodes_dict[id(vensor)]
            temp = curr_node.grad1*grads
            if(curr_node.vensor1.is_leaf):
                curr_node.vensor1.grad += temp
            cls.backward(temp, curr_node.vensor1) 
          

            if(curr_node.vensor2 != None):
                temp = curr_node.grad2*grads
                if(curr_node.vensor2.is_leaf):
                    curr_node.vensor2.grad += temp
                cls.backward(temp, curr_node.vensor2)

    
