"""
Impurity criteria for decision tree splits.
"""

import numpy as np

def calculate_impurity(y, criterion='gini'):
    """
    Calculate the impurity of a node.
    
    Parameters:
    -----------
    y : array-like
        Target vector
    criterion : str
        Impurity criterion ('gini' or 'entropy')
        
    Returns:
    --------
    impurity : float
        Node impurity
    """
    # Get class probabilities
    n_samples = len(y)
    if n_samples == 0:
        return 0
        
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / n_samples
    
    if criterion == 'gini':
        # Gini impurity
        return 1 - np.sum(probabilities ** 2)
    else:
        # Entropy
        # Avoid log(0) errors
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))