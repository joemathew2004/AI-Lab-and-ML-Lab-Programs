import pandas as pd
import numpy as np

# Load the dataset
filename = 'PlayTennis.csv'
data = pd.read_csv(filename)

# Prepare features and target variable
X = data.drop('Play Tennis', axis=1)
y = data['Play Tennis']

def entropy(y):
    """Calculate the entropy of a label array."""
    labels, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def information_gain(X, y, feature):
    """Calculate the information gain of a feature."""
    total_entropy = entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / counts.sum()) * entropy(y[X[feature] == values[i]])
                               for i in range(len(values))])
    return total_entropy - weighted_entropy

class DecisionTreeNode:
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label

    def print_tree(self, depth=0):
        """Recursively print the decision tree."""
        indent = "  " * depth
        if self.label is not None:
            print(f"{indent}Leaf Node: Class = {self.label}")
        else:
            print(f"{indent}Node: {self.feature} = {self.value}")
            if self.left:
                print(f"{indent}  -> Left:")
                self.left.print_tree(depth + 1)
            if self.right:
                print(f"{indent}  -> Right:")
                self.right.print_tree(depth + 1)

def build_tree(X, y):
    """Recursively build a decision tree."""
    if len(np.unique(y)) == 1:
        return DecisionTreeNode(label=y.iloc[0])
    
    if X.empty:
        return DecisionTreeNode(label=y.mode()[0])
    
    best_feature = max(X.columns, key=lambda feature: information_gain(X, y, feature))
    node = DecisionTreeNode(feature=best_feature)
    
    for value in np.unique(X[best_feature]):
        subset_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        subset_y = y[X[best_feature] == value]
        child_node = build_tree(subset_X, subset_y)
        if node.left is None:
            node.left = child_node
            node.value = value
        else:
            node.right = child_node
    
    return node

def predict(tree, sample):
    """Predict the class of a sample."""
    if tree.label is not None:
        return tree.label
    
    feature_value = sample[tree.feature]
    if feature_value == tree.value:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)

def get_user_input():
    print("Enter values for Outlook, Temperature, Humidity & Wind:")
    outlook = input("Outlook (Sunny, Overcast, Rain): ")
    temperature = input("Temperature (Hot, Mild, Cool): ")
    humidity = input("Humidity (High, Normal): ")
    wind = input("Wind (Weak, Strong): ")
    return [outlook, temperature, humidity, wind]

# Build the decision tree
tree = build_tree(X, y)

# Print the decision tree
print("Decision Tree Structure:")
tree.print_tree()

# Convert user input to DataFrame
user_input = get_user_input()
user_df = pd.DataFrame([user_input], columns=X.columns)

# Predict the class for the user input
prediction = predict(tree, user_df.iloc[0])
print(f"Predicted class: {prediction}")
