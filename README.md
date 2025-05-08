# Categorical Decision Tree

A Python implementation of decision trees that directly handles categorical features without binarization, improving interpretability and potentially performance.

## Overview

This implementation allows decision trees to create multi-way splits directly on categorical features, eliminating the need for one-hot encoding or other binarization techniques. The design follows scikit-learn's API conventions for seamless integration with existing machine learning pipelines.

## Features

- **Native Categorical Support**: Create direct multi-way splits on categorical features
- **Improved Interpretability**: More intuitive tree decisions (e.g., "if color is red" instead of "if color_red is 1")
- **Scikit-learn Compatible API**: Familiar fit/predict interface for easy integration
- **Automatic Feature Type Detection**: Intelligently identifies categorical vs. numerical features
- **Multiple Impurity Measures**: Supports both Gini impurity and information gain (entropy)
- **Feature Importance**: Quantifies feature contributions similar to scikit-learn
- **Rich Visualization**: Text-based and graphical tree visualization tools




## Usage

Basic example:

```python
from categorical_tree import CategoricalDecisionTree
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train the model
tree = CategoricalDecisionTree(
    criterion='entropy',
    max_depth=5,
    feature_types=['categorical', 'numerical', 'categorical']  # Optional
)
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)
probabilities = tree.predict_proba(X_test)

# Visualize the tree
from categorical_tree.visualization import export_text, display_tree

print(export_text(tree))
display_tree(tree, class_names=['Class1', 'Class2'])
```
## Requirements

- NumPy
- Pandas
- Graphviz (for visualization)
- IPython (for notebook visualizations)

## Advantages Over Standard Decision Trees

1. No Information Loss: Avoids information loss that can occur during binarization
1. Smaller Trees: Often produces more compact trees with fewer nodes
1. Clearer Interpretability: Direct mapping to original feature values
1. Better Handling of High-Cardinality Features: More efficient representation

## How It Works

Unlike standard decision trees that convert categorical features to binary indicators, this implementation:

1. Directly partitions the data based on categorical values
1. Creates multi-way splits (one branch per category or group of categories)
1. Calculates impurity measures (Gini/entropy) adapted for multi-way splits
1. Prevents the same feature from being used multiple times along the same path

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
