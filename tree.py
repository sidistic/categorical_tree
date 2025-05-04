"""
Main decision tree classifier implementation.
"""

import numpy as np
import pandas as pd

from .node import Node
from .criteria import calculate_impurity
from .splitter import find_best_split
from .utils import infer_feature_types

class CategoricalDecisionTree:
    """
    Decision tree classifier that handles categorical features natively.
    
    Parameters:
    -----------
    criterion : str, default='gini'
        Function to measure the quality of a split. Supported criteria are
        'gini' for the Gini impurity and 'entropy' for the information gain.
        
    max_depth : int, default=None
        Maximum depth of the tree. If None, nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.
        
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
        
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
        
    feature_types : list or None, default=None
        List indicating feature types ('categorical' or 'numerical').
        If None, all features are treated as 'numerical' except for object or
        categorical dtypes which are treated as 'categorical'.
    """
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, feature_types=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_types = feature_types
        self.root = None
        self.n_classes_ = None
        self.classes_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns:
        --------
        self : object
        """
        # Convert input to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Determine feature types if not provided
        if self.feature_types is None:
            self.feature_types = infer_feature_types(X)
        
        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Map classes to integers
        y_encoded = np.zeros(y.shape, dtype=int)
        for i, cls in enumerate(self.classes_):
            y_encoded[y == cls] = i
        
        # Build the tree
        from .splitter import build_tree  # Import here to avoid circular import
        self.root = build_tree(
            X, y_encoded, self.feature_types, self.max_depth,
            self.min_samples_split, self.min_samples_leaf, self.criterion
        )
        
        # Calculate feature importances
        self._calculate_feature_importances(X.shape[1])
        
        return self
    
    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        X = np.asarray(X)
        y_pred_indices = np.zeros(X.shape[0], dtype=int)
        
        for i, sample in enumerate(X):
            node = self.root
            while not node.is_leaf:
                feature_idx = node.feature_index
                feature_value = sample[feature_idx]
                
                if node.feature_type == 'categorical':
                    # Handle missing or unseen categories
                    if feature_value not in node.category_map:
                        # Default to the most common category
                        child_idx = max(node.category_map.values(), key=list(node.category_map.values()).count)
                    else:
                        child_idx = node.category_map[feature_value]
                    
                    if child_idx < len(node.children):
                        node = node.children[child_idx]
                    else:
                        # Fallback to leaf
                        break
                else:
                    # Numerical feature
                    if feature_value <= node.threshold:
                        if len(node.children) > 0:
                            node = node.children[0]  # Left child
                        else:
                            break
                    else:
                        if len(node.children) > 1:
                            node = node.children[1]  # Right child
                        else:
                            break
            
            y_pred_indices[i] = node.prediction
        
        # Convert indices back to original class labels
        return self.classes_[y_pred_indices]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        proba : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        X = np.asarray(X)
        proba = np.zeros((X.shape[0], self.n_classes_))
        
        for i, sample in enumerate(X):
            node = self.root
            while not node.is_leaf:
                feature_idx = node.feature_index
                feature_value = sample[feature_idx]
                
                if node.feature_type == 'categorical':
                    if feature_value not in node.category_map:
                        # Default to the most common category
                        child_idx = max(node.category_map.values(), key=list(node.category_map.values()).count)
                    else:
                        child_idx = node.category_map[feature_value]
                    
                    if child_idx < len(node.children):
                        node = node.children[child_idx]
                    else:
                        break
                else:
                    if feature_value <= node.threshold:
                        if len(node.children) > 0:
                            node = node.children[0]
                        else:
                            break
                    else:
                        if len(node.children) > 1:
                            node = node.children[1]
                        else:
                            break
            
            # Calculate class probabilities from distribution
            proba[i] = node.class_distribution / np.sum(node.class_distribution)
        
        return proba
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns:
        --------
        score : float
            Mean accuracy.
        """
        return np.mean(self.predict(X) == y)
    
    def _calculate_feature_importances(self, n_features):
        """
        Calculate feature importances based on impurity reduction.
        
        Parameters:
        -----------
        n_features : int
            Number of features
        """
        self.feature_importances_ = np.zeros(n_features)
        
        def _collect_importances(node, total_samples):
            if node.is_leaf:
                return
            
            # Calculate importance for this split
            feature_idx = node.feature_index
            impurity_decrease = node.impurity * node.samples
            
            for child in node.children:
                impurity_decrease -= child.impurity * child.samples
            
            self.feature_importances_[feature_idx] += impurity_decrease / total_samples
            
            # Recurse on children
            for child in node.children:
                _collect_importances(child, total_samples)
        
        _collect_importances(self.root, self.root.samples)
        
        # Normalize
        if np.any(self.feature_importances_):
            self.feature_importances_ /= np.sum(self.feature_importances_)