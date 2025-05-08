"""
Basic usage example for the CategoricalDecisionTree.
"""

import numpy as np
import pandas as pd
from categorical_tree import CategoricalDecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a sample dataset with mixed feature types
np.random.seed(42)

# Numerical features
age = np.random.randint(18, 70, 100)
income = np.random.randint(20000, 100000, 100)

# Categorical features
education = np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], 100)
marital_status = np.random.choice(['single', 'married', 'divorced'], 100)

# Create target: buy a product or not
# Rule: People with masters/phd OR (age < 30 and income > 50000) are more likely to buy
y = np.zeros(100)
for i in range(100):
    if education[i] in ['masters', 'phd']:
        y[i] = 1
    elif age[i] < 30 and income[i] > 50000:
        y[i] = 1
    elif marital_status[i] == 'married' and income[i] > 70000:
        y[i] = 1
    else:
        # Add some randomness
        y[i] = np.random.choice([0, 1], p=[0.8, 0.2])

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'education': education,
    'marital_status': marital_status,
    'buy': y.astype(int)
})

# Split the data
X = df.drop('buy', axis=1)
y = df['buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
feature_types = ['numerical', 'numerical', 'categorical', 'categorical']
tree = CategoricalDecisionTree(criterion='entropy', max_depth=3, feature_types=feature_types)
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the tree
from categorical_tree.visualization import export_text, display_tree
print(export_text(tree))

# Feature importances
for i, importance in enumerate(tree.feature_importances_):
    print(f"Feature {X.columns[i]}: {importance:.4f}")

display_tree(tree, save_path= "example.png")

