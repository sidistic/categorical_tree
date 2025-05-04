"""
Node definitions for the decision tree.
"""

class Node:
    """
    Tree node class for the decision tree.
    
    Attributes:
    -----------
    feature_index : int or None
        Index of feature to split on
    feature_type : str or None
        Type of feature ('categorical' or 'numerical')
    threshold : float or None
        Threshold for numerical features
    category_map : dict
        For categorical features: {value: child_index}
    children : list
        Child nodes
    is_leaf : bool
        Whether this is a leaf node
    prediction : int or None
        Class prediction for leaf nodes
    impurity : float
        Node impurity (Gini or entropy)
    samples : int
        Number of samples at this node
    class_distribution : array
        Distribution of classes at this node
    """
    
    def __init__(self):
        self.feature_index = None        # Index of feature to split on
        self.feature_type = None         # 'categorical' or 'numerical'
        self.threshold = None            # For numerical features
        self.category_map = {}           # For categorical features: {value: child_index}
        self.children = []               # Child nodes (multiple for categorical features)
        self.is_leaf = False             # Whether this is a leaf node
        self.prediction = None           # Class prediction for leaf nodes
        self.impurity = None             # Node impurity (Gini or entropy)
        self.samples = None              # Number of samples at this node
        self.class_distribution = None   # Distribution of classes at this node