import numpy as np

class Operation():
    """
    An Operation is a node in a "Graph". TensorFlow will also use this concept of a Graph.
    
    This Operation class will be inherited by other classes that actually compute the specific
    operation, such as adding or matrix multiplication.
    """
    
    def __init__(self, input_nodes = []):
        """
        Intialize an Operation
        """
        self.input_nodes = input_nodes # The list of input nodes
        self.output_nodes = [] # List of nodes consuming this node's output
        
        # For every node in the input, we append this operation (self) to the list of
        # the consumers of the input nodes
        for node in input_nodes:
            node.output_nodes.append(self)
        
        # There will be a global default graph (TensorFlow works this way)
        # We will then append this particular operation
        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)
  
    def compute(self):
        """ 
        This is a placeholder function. It will be overwritten by the actual specific operation
        that inherits from this class.
        
        """
        
        pass

class add(Operation):
    
    def __init__(self, x, y):
         
        super().__init__([x, y])

    def compute(self, x_var, y_var):
         
        self.inputs = [x_var, y_var]
        return x_var + y_var

    def __repr__(self):
        return 'add(input_nodes=%s)' % (str(self.input_nodes))

class multiply(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a_var, b_var):
         
        self.inputs = [a_var, b_var]
        return a_var * b_var

    def __repr__(self):
        return 'multiply(input_nodes=[%s])' % (', '.join(map(str,self.input_nodes)))

class matmul(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a_mat, b_mat):
         
        self.inputs = [a_mat, b_mat]
        return a_mat.dot(b_mat)

class Placeholder():
    """
    A placeholder is a node that needs to be provided a value for computing the output in the Graph.
    占位符是一个节点，为计算图的输出提供值。
    """
    
    def __init__(self):
        
        self.output_nodes = []
        
        _default_graph.placeholders.append(self)

    def __repr__(self):
        return 'Placeholder()'

class Variable():
    """
    This variable is a changeable parameter of the Graph.
    是计算图的可更改参数。
    """
    
    def __init__(self, initial_value = None):
        
        self.value = initial_value
        self.output_nodes = []
        
         
        _default_graph.variables.append(self)

    def __repr__(self):
        return 'Variable(value=%s)' % (str(self.value))


class Graph():
    
    
    def __init__(self):
        
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def set_as_default(self):
        """
        Sets this Graph instance as the Global Default Graph
        """
        global _default_graph
        _default_graph = self

# Traversing Operation Nodes
def traverse_postorder(operation):
    """ 
    PostOrder Traversal of Nodes. Basically makes sure computations are done in 
    the correct order (Ax first , then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the basic fundamentals of deep learning.
    
    用于确保计算按照正确的顺序进行。
    """
    
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder

class Session:
    
    def run(self, operation, feed_dict = {}):
        """ 
          operation: 要计算的操作
          feed_dict: 占位符和输入值的映射（Dictionary mapping placeholders to input values (the data) ）
        """
        
        # 以正确的舒徐放置节点（Puts nodes in correct order）
        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:

            if type(node) == Placeholder:
                node.output = feed_dict[node]
                
            elif type(node) == Variable:
                node.output = node.value
                
            else: # Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
                
            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        
        # Return the requested node value
        return operation.output

    def run_debug(self, operation, feed_dict = {}):
        """ 
          operation: 要计算的操作
          feed_dict: 占位符和输入值的映射（Dictionary mapping placeholders to input values (the data) ）
        """
        
        # 以正确的顺序放置节点（Puts nodes in correct order）
        nodes_postorder = traverse_postorder(operation)
        print('type of nodes_postorder:', type(nodes_postorder), '\nnodes_postorder:\n', nodes_postorder)
        return 'shit'
        
        for node in nodes_postorder:

            if type(node) == Placeholder:
                node.output = feed_dict[node]
                
            elif type(node) == Variable:
                node.output = node.value
                
            else: # Operation
                
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
                
            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        
        # Return the requested node value
        return operation.output

g = Graph()
g.set_as_default()
A = Variable(10)
b = Variable(1)

# Will be filled out later
x = Placeholder()

y = multiply(A, x)
z = add(y,b)

sess = Session()
result = sess.run_debug(operation=z,feed_dict={x:10})
# print("type(result):", type(result), " result:", result)