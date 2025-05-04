# Categorical Decision Tree

A custom implementation of decision trees that natively handles categorical features without binarization.

## Features

- Direct handling of categorical variables with multi-way splits
- Compatible API with scikit-learn
- Improved tree readability
- Support for both Gini impurity and information gain criteria
- Automatic feature type detection
- Feature importance calculation

## Installation

```bash
pip install -e .
```

## Usage

Basic example:

```python
from categorical_tree import CategoricalDecisionTree
from sklearn.model_selection import train_test_split

# Prepare your data with mixed categorical and numerical features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train the model
tree = CategoricalDecisionTree(criterion='entropy', max_depth=5)
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)

# Get class probabilities
probas = tree.predict_proba(X_test)

# Visualize the tree
from categorical_tree.visualization import export_text
print(export_text(tree))
```

## Documentation

See the examples directory for more usage examples.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
