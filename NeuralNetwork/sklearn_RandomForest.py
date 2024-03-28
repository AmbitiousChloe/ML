import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv("ModifiedData.csv")  # Adjust the path to your dataset

# Assuming 'Label' is your target variable
X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It's a good practice to scale your data, although it's not always necessary for Random Forest
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Random Forest model
# n_estimators is the number of trees in the forest
# random_state is set for reproducibility of results
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
trees = rf_model.estimators_

# Example: Analyzing the first tree
first_tree = trees[0]

# Get the tree's properties
tree_structure = first_tree.tree_

# Basic properties
print(f"Number of nodes in the first tree: {tree_structure.node_count}")
print(f"Maximum depth of the first tree: {tree_structure.max_depth}")

# You could also plot the tree. Here, we limit the depth for readability
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(first_tree, filled=True, rounded=True, feature_names=X.columns, max_depth=3, fontsize=10)
plt.show()



