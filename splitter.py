"""
Split finding algorithms for decision trees.
"""

import numpy as np
from .criteria import calculate_impurity
from .node import Node

def find_best_split(X, y, feature_types, criterion='gini', used_features=None):
    """
    Find the best feature to split on, excluding already used features.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_types : list
        List indicating feature types ('categorical' or 'numerical')
    criterion : str
        Splitting criterion ('gini' or 'entropy')
    used_features : set
        Set of features to exclude from consideration
        
    Returns:
    --------
    best_feature : int
        Index of best feature to split on
    best_split_info : dict
        Information about the best split
    """
    if used_features is None:
        used_features = set()
        
    best_gain = -float('inf')
    best_feature = None
    best_split_info = None
    
    for feature_idx in range(X.shape[1]):
        # Skip already used features
        if feature_idx in used_features:
            continue
            
        if feature_types[feature_idx] == 'categorical':
            # Handle categorical feature
            gain, split_info = find_best_categorical_split(X, y, feature_idx, criterion)
        else:
            # Handle numerical feature
            gain, split_info = find_best_numerical_split(X, y, feature_idx, criterion)
            
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_idx
            best_split_info = split_info
            
    return best_feature, best_split_info

def find_best_categorical_split(X, y, feature_idx, criterion='gini'):
    """
    Find the best categorical split for a given feature.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_idx : int
        Index of the feature to evaluate
    criterion : str
        Splitting criterion ('gini' or 'entropy')
        
    Returns:
    --------
    gain : float
        Information gain or Gini gain for the best split
    split_info : dict
        Information about the best split
    """
    # Get unique values for the feature
    unique_values = np.unique(X[:, feature_idx])
    
    # Calculate parent impurity
    parent_impurity = calculate_impurity(y, criterion)
    
    # Multi-way split for all unique values
    category_map = {}
    child_indices = {}
    
    for value in unique_values:
        mask = X[:, feature_idx] == value
        child_indices[value] = np.where(mask)[0]
    
    # Calculate weighted impurity of children
    weighted_child_impurity = 0
    for value, indices in child_indices.items():
        if len(indices) == 0:
            continue
            
        child_y = y[indices]
        weight = len(indices) / len(y)
        child_impurity = calculate_impurity(child_y, criterion)
        weighted_child_impurity += weight * child_impurity
        category_map[value] = indices
    
    # Calculate gain
    gain = parent_impurity - weighted_child_impurity
    
    return gain, {
        'type': 'categorical',
        'category_map': category_map,
        'child_indices': child_indices
    }

def find_best_numerical_split(X, y, feature_idx, criterion='gini'):
    """
    Find the best numerical split for a given feature.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_idx : int
        Index of the feature to evaluate
    criterion : str
        Splitting criterion ('gini' or 'entropy')
        
    Returns:
    --------
    gain : float
        Information gain or Gini gain for the best split
    split_info : dict
        Information about the best split
    """
    # Get values for the feature
    values = X[:, feature_idx]
    
    # Sort values and corresponding targets
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_y = y[sorted_indices]
    
    # Calculate parent impurity
    parent_impurity = calculate_impurity(sorted_y, criterion)
    
    # Find potential split points
    unique_values = np.unique(sorted_values)
    if len(unique_values) <= 1:
        return -float('inf'), None
    
    split_points = (unique_values[:-1] + unique_values[1:]) / 2
    
    # Find best split point
    best_gain = -float('inf')
    best_threshold = None
    best_left_indices = None
    best_right_indices = None
    
    # For each potential split point
    for threshold in split_points:
        left_mask = values <= threshold
        right_mask = ~left_mask
        
        left_y = y[left_mask]
        right_y = y[right_mask]
        
        # Skip if either side is empty
        if len(left_y) == 0 or len(right_y) == 0:
            continue
        
        # Calculate weighted impurity
        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        
        left_impurity = calculate_impurity(left_y, criterion)
        right_impurity = calculate_impurity(right_y, criterion)
        
        weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
        
        # Calculate gain
        gain = parent_impurity - weighted_impurity
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_left_indices = np.where(left_mask)[0]
            best_right_indices = np.where(right_mask)[0]
    
    if best_threshold is None:
        return -float('inf'), None
    
    return best_gain, {
        'type': 'numerical',
        'threshold': best_threshold,
        'left_indices': best_left_indices,
        'right_indices': best_right_indices
    }

def build_tree(X, y, feature_types, max_depth=None, min_samples_split=2, 
               min_samples_leaf=1, criterion='gini', depth=0, used_features=None):
    """
    Recursively build the decision tree.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_types : list
        List indicating feature types ('categorical' or 'numerical')
    max_depth : int or None
        Maximum depth of the tree
    min_samples_split : int
        Minimum samples required to split a node
    min_samples_leaf : int
        Minimum samples required in a leaf node
    criterion : str
        Splitting criterion ('gini' or 'entropy')
    depth : int
        Current depth of the tree
    used_features : set
        Set of features already used in the current path
        
    Returns:
    --------
    node : Node
        Root node of the subtree
    """
    # Initialize used_features if not provided
    if used_features is None:
        used_features = set()
    
    node = Node()
    node.samples = len(y)
    node.class_distribution = np.bincount(y, minlength=np.max(y)+1)
    node.impurity = calculate_impurity(y, criterion)
    
    # Check stopping criteria
    if (max_depth is not None and depth >= max_depth) or \
       len(y) < min_samples_split or \
       len(np.unique(y)) == 1 or \
       len(used_features) == X.shape[1]:  # All features used
        node.is_leaf = True
        node.prediction = np.argmax(node.class_distribution)
        return node
    
    # Find best split using only unused features
    feature_idx, split_info = find_best_split(X, y, feature_types, 
                                                             criterion, used_features)
    
    if feature_idx is None or split_info is None:
        node.is_leaf = True
        node.prediction = np.argmax(node.class_distribution)
        return node
    
    node.feature_index = feature_idx
    node.feature_type = feature_types[feature_idx]
    
    # Add this feature to used_features for this path
    path_used_features = used_features.copy()
    path_used_features.add(feature_idx)
    
    if split_info['type'] == 'categorical':
        # Handle categorical split
        node.category_map = {}
        
        for value, indices in split_info['child_indices'].items():
            if len(indices) < min_samples_leaf:
                continue
                
            child_X = X[indices]
            child_y = y[indices]
            
            child_node = build_tree(
                child_X, child_y, feature_types, max_depth, 
                min_samples_split, min_samples_leaf, criterion, depth+1,
                path_used_features  # Pass the updated used_features
            )
            
            node.children.append(child_node)
            node.category_map[value] = len(node.children) - 1
    else:
        # Handle numerical split
        node.threshold = split_info['threshold']
        left_indices = split_info['left_indices']
        right_indices = split_info['right_indices']
        
        # Build left subtree
        if len(left_indices) >= min_samples_leaf:
            left_child = build_tree(
                X[left_indices], y[left_indices], feature_types, 
                max_depth, min_samples_split, min_samples_leaf, criterion, depth+1,
                path_used_features  # Pass the updated used_features
            )
            node.children.append(left_child)
        
        # Build right subtree
        if len(right_indices) >= min_samples_leaf:
            right_child = build_tree(
                X[right_indices], y[right_indices], feature_types, 
                max_depth, min_samples_split, min_samples_leaf, criterion, depth+1,
                path_used_features  # Pass the updated used_features
            )
            node.children.append(right_child)
    
    return node

def validate_tree(node, depth=0, max_depth=100):
    """
    Validate the tree structure to catch potential errors.
    
    Parameters:
    -----------
    node : Node
        Root node to validate
    depth : int
        Current depth
    max_depth : int
        Maximum allowed depth to prevent infinite recursion
        
    Returns:
    --------
    issues : list
        List of identified issues
    """
    issues = []
    
    if depth > max_depth:
        issues.append(f"Tree exceeds maximum depth of {max_depth}, possible circular reference")
        return issues
    
    if node is None:
        issues.append("Null node encountered")
        return issues
    
    # Check for invalid feature index
    if not node.is_leaf and node.feature_index is None:
        issues.append("Non-leaf node with missing feature index")
    
    # Check for categorical features
    if node.feature_type == 'categorical':
        # Check for empty category map
        if not node.category_map:
            issues.append("Categorical node with empty category map")
        
        # Check child indices are valid
        for value, child_idx in node.category_map.items():
            if child_idx >= len(node.children):
                issues.append(f"Category map points to non-existent child index {child_idx}")
    
    # Recursively validate children
    for i, child in enumerate(node.children):
        child_issues = validate_tree(child, depth + 1, max_depth)
        issues.extend([f"Child {i}: {issue}" for issue in child_issues])
    
    return issues