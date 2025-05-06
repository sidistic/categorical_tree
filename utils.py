"""
Utility functions for the decision tree.
"""

import numpy as np
import pandas as pd

def infer_feature_types(X):
    """
    Infer feature types from the data.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
        
    Returns:
    --------
    feature_types : list
        List indicating feature types ('categorical' or 'numerical')
    """
    feature_types = []
    
    # Try to infer from pandas DataFrame
    if hasattr(X, 'dtypes'):
        for dtype in X.dtypes:
            if dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
                feature_types.append('categorical')
            else:
                feature_types.append('numerical')
    else:
        # Infer from numpy array
        for j in range(X.shape[1]):
            column = X[:, j]
            # Check if the column contains string data
            if isinstance(column[0], str):
                feature_types.append('categorical')
            # Check if the column has a small number of unique values
            elif len(np.unique(column)) < 0.05 * len(column):
                feature_types.append('categorical')
            else:
                feature_types.append('numerical')
    
    return feature_types

def count_depth(node):
    """Count the maximum depth of the tree from a given node."""
    if node is None or node.is_leaf:
        return 0
    return 1 + max([count_depth(child) for child in node.children] or [0])