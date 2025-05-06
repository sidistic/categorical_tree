"""
Visualization utilities for decision trees.
"""
import numpy as np
from graphviz import Source
from IPython.display import display as ipython_display

def export_text(tree):
    """
    Export the decision tree in a readable text format without repetition.
    """
    if tree.root is None:
        return "Tree not fitted yet."
    
    lines = []
    
    def _traverse_tree(node, depth=0):
        indent = "    " * depth
        
        if node.is_leaf:
            class_counts = [f"{count} samples of class {tree.classes_[i]}" 
                           for i, count in enumerate(node.class_distribution) if count > 0]
            lines.append(f"{indent}Predict {tree.classes_[node.prediction]} "
                         f"({', '.join(class_counts)})")
            return
        
        feature_idx = node.feature_index
        feature_name = tree.feature_names[feature_idx]
        
        if node.feature_type == 'categorical':
            # Group children by category for better readability
            category_groups = {}
            for value, child_idx in node.category_map.items():
                if child_idx < len(node.children):
                    if child_idx not in category_groups:
                        category_groups[child_idx] = []
                    category_groups[child_idx].append(value)
            
            for child_idx, values in category_groups.items():
                if child_idx < len(node.children):
                    child = node.children[child_idx]
                    values_str = ", ".join([str(v) for v in values])
                    lines.append(f"{indent}If {feature_name} in [{values_str}]: ({child.samples} samples)")
                    _traverse_tree(child, depth + 1)
        else:
            # Numerical split
            if len(node.children) > 0:
                left_child = node.children[0]
                lines.append(f"{indent}If {feature_name} <= {node.threshold:.4f}: ({left_child.samples} samples)")
                _traverse_tree(left_child, depth + 1)
            
            if len(node.children) > 1:
                right_child = node.children[1]
                lines.append(f"{indent}If {feature_name} > {node.threshold:.4f}: ({right_child.samples} samples)")
                _traverse_tree(right_child, depth + 1)
    
    # Start the recursion from the root
    _traverse_tree(tree.root)
    
    return "\n".join(lines)

def export_graphviz(tree, out_file=None, feature_names=None, class_names=None, filled=True):
    """
    Export a decision tree in DOT format.
    
    Parameters:
    -----------
    tree : CategoricalDecisionTree
        The decision tree to be exported.
    out_file : file object or str, optional
        Handle or name of the output file.
    feature_names : list of str, optional
        Names of each of the features.
    class_names : list of str, optional
        Names of each of the target classes.
    filled : bool, optional
        When True, paint nodes to indicate majority class.
        
    Returns:
    --------
    dot_data : str
        String representation of the input tree in DOT format.
    """
    if tree.root is None:
        return "digraph G {}"
    
    # Use tree's feature_names if not provided
    if feature_names is None:
        feature_names = tree.feature_names
    
    # Use tree's class_names if not provided
    if class_names is None:
        class_names = [str(c) for c in tree.classes_]
    
    # Colors for nodes
    colors = ['#ffffff', '#ebf5ff', '#cce7ff', '#a6d4ff', '#75bbff', '#51a8ff', '#2b94ff', '#0a7fff', '#0066cc']
    
    dot_data = ['digraph Tree {']
    dot_data.append('  node [shape=box, style="filled, rounded", color="black", fontname=helvetica];')
    dot_data.append('  edge [fontname=helvetica];')
    
    # Node counter for unique IDs
    node_ids = {}
    next_id = 0
    
    # Function to assign unique IDs to nodes
    def get_node_id(node):
        nonlocal next_id
        if node not in node_ids:
            node_ids[node] = next_id
            next_id += 1
        return node_ids[node]
    
    # Function to traverse tree and add nodes to DOT graph
    def add_nodes_edges(node):
        if node is None:
            return
        
        node_id = get_node_id(node)
        
        if node.is_leaf:
            # Format for leaf nodes
            class_idx = node.prediction
            class_name = class_names[class_idx]
            
            # Calculate proportions for class distribution
            total_samples = node.samples
            class_proportions = node.class_distribution / total_samples if total_samples > 0 else []
            
            # Format class distribution
            class_dist_str = ', '.join([f"{class_names[i]}: {count}" for i, count in enumerate(node.class_distribution) if count > 0])
            
            # Color based on class purity
            color = colors[min(int(class_proportions[class_idx] * 8), 8)] if filled and len(class_proportions) > class_idx else '#ffffff'
            
            label = f'Predict: {class_name}\\nSamples: {total_samples}\\nClass Distribution: {class_dist_str}'
            dot_data.append(f'  {node_id} [label="{label}", fillcolor="{color}"];')
        else:
            # Format for decision nodes
            feature_idx = node.feature_index
            feature_name = feature_names[feature_idx]
            
            # Calculate class proportions for coloring
            total_samples = node.samples
            majority_class = np.argmax(node.class_distribution)
            class_proportions = node.class_distribution / total_samples if total_samples > 0 else []
            
            # Color based on majority class proportion
            color = colors[min(int(class_proportions[majority_class] * 8), 8)] if filled and len(class_proportions) > 0 else '#ffffff'
            
            # Format class distribution
            class_dist_str = ', '.join([f"{class_names[i]}: {count}" for i, count in enumerate(node.class_distribution) if count > 0])
            
            if node.feature_type == 'categorical':
                # Show categorical mapping
                mapping_str = '\\n'.join([f'{value} → child {idx}' for value, idx in node.category_map.items()])
                label = f'{feature_name}\\nImpurity: {node.impurity:.4f}\\nSamples: {total_samples}\\nClass Distribution: {class_dist_str}\\n\\nCategories:\\n{mapping_str}'
                dot_data.append(f'  {node_id} [label="{label}", fillcolor="{color}"];')
                
                # Add edges to children
                for value, child_idx in node.category_map.items():
                    if child_idx < len(node.children):
                        child = node.children[child_idx]
                        child_id = get_node_id(child)
                        dot_data.append(f'  {node_id} -> {child_id} [label="{value}"];')
                        
                        # Recursively add child nodes
                        add_nodes_edges(child)
            else:
                # For numerical features
                label = f'{feature_name} <= {node.threshold:.4f}\\nImpurity: {node.impurity:.4f}\\nSamples: {total_samples}\\nClass Distribution: {class_dist_str}'
                dot_data.append(f'  {node_id} [label="{label}", fillcolor="{color}"];')
                
                # Add edges to children
                if len(node.children) > 0:
                    child = node.children[0]
                    child_id = get_node_id(child)
                    dot_data.append(f'  {node_id} -> {child_id} [label="True"];')
                    add_nodes_edges(child)
                
                if len(node.children) > 1:
                    child = node.children[1]
                    child_id = get_node_id(child)
                    dot_data.append(f'  {node_id} -> {child_id} [label="False"];')
                    add_nodes_edges(child)
    
    # Traverse the tree and add all nodes/edges
    add_nodes_edges(tree.root)
    
    dot_data.append('}')
    dot_data = '\n'.join(dot_data)
    
    # Write to file if specified
    if out_file is not None:
        if isinstance(out_file, str):
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(dot_data)
        else:
            out_file.write(dot_data)
    
    return dot_data

def display_tree(tree, class_names=None, display_in_notebook=True, 
                 save_path=None, format='png', view=False):
    """
    Display and/or save a graphical representation of the decision tree.
    
    Parameters:
    -----------
    tree : CategoricalDecisionTree
        The decision tree to visualize
    class_names : list of str, optional
        Names of the target classes
    display_in_notebook : bool, default=True
        Whether to display the tree in the notebook (only works in Jupyter/IPython)
    save_path : str or None, default=None
        Path to save the visualization (without extension)
        If None, the image will not be saved
    format : str, default='png'
        File format to save the visualization ('png', 'pdf', 'svg', etc.)
    view : bool, default=False
        Whether to open the rendered image in the default viewer
        
    Returns:
    --------
    src : graphviz.Source
        The Source object containing the tree visualization
    
    Notes:
    ------
    This function requires graphviz to be installed on your system.
    Install it with:
    - On Ubuntu/Debian: apt-get install graphviz
    - On macOS: brew install graphviz
    - On Windows: download and install from https://graphviz.org/download/
    """
    # Get the DOT representation
    dot_data = export_graphviz(tree, class_names=class_names)
    
    # Create a Source object
    src = Source(dot_data)
    
    # Display in notebook if requested
    if display_in_notebook:
        try:
            ipython_display(src)
        except Exception as e:
            print(f"Warning: Could not display the graph in the notebook. {e}")
            print("Make sure you're running in a Jupyter/IPython environment.")
    
    # Save to file if a path is provided
    if save_path is not None:
        try:
            src.render(filename=save_path, format=format, cleanup=True, view=view)
            print(f"Tree visualization saved to {save_path}.{format}")
        except Exception as e:
            print(f"Error saving tree visualization: {e}")
            print("Make sure graphviz is installed on your system.")
    
    return src