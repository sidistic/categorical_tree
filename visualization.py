"""
Visualization utilities for decision trees.
"""

def export_text(tree):
    """
    Export the decision tree in a readable text format.
    
    Parameters:
    -----------
    tree : CategoricalDecisionTree
        The fitted decision tree
        
    Returns:
    --------
    text : str
        Text representation of the decision tree
    """
    if tree.root is None:
        return "Tree not fitted yet."
    
    def _get_text_representation(node, depth=0, path_annotation=""):
        indent = "    " * depth
        result = []
        
        if node.is_leaf:
            class_counts = [f"{count} samples of class {tree.classes_[i]}" 
                           for i, count in enumerate(node.class_distribution) if count > 0]
            result.append(f"{indent}{path_annotation}Predict {tree.classes_[node.prediction]} "
                         f"({', '.join(class_counts)})")
            return result
        
        feature_idx = node.feature_index
        
        if node.feature_type == 'categorical':
            # Group children by category for better readability
            category_groups = {}
            for value, child_idx in node.category_map.items():
                if child_idx < len(node.children):
                    if child_idx not in category_groups:
                        category_groups[child_idx] = []
                    category_groups[child_idx].append(value)
            
            for child_idx, values in category_groups.items():
                child = node.children[child_idx]
                values_str = ", ".join([str(v) for v in values])
                condition = f"Feature {feature_idx} in [{values_str}]"
                result.append(f"{indent}{path_annotation}If {condition}:")
                result.extend(_get_text_representation(
                    child, depth + 1, 
                    path_annotation=f"({len(child.class_distribution)} samples) "
                ))
        else:
            # Numerical split
            if len(node.children) > 0:
                condition = f"Feature {feature_idx} <= {node.threshold:.4f}"
                result.append(f"{indent}{path_annotation}If {condition}:")
                result.extend(_get_text_representation(
                    node.children[0], depth + 1,
                    path_annotation=f"({node.children[0].samples} samples) "
                ))
            
            if len(node.children) > 1:
                condition = f"Feature {feature_idx} > {node.threshold:.4f}"
                result.append(f"{indent}{path_annotation}If {condition}:")
                result.extend(_get_text_representation(
                    node.children[1], depth + 1,
                    path_annotation=f"({node.children[1].samples} samples) "
                ))
        
        return result
    
    lines = _get_text_representation(tree.root)
    return "\n".join(lines)