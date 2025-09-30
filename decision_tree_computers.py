import pandas as pd
import numpy as np

train_data_m = pd.read_csv("Buy_Computer.csv")
train_data_m = train_data_m.drop(columns=['id'])   # Drop the 'id' column

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entr = 0
   
    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row)
        total_entr += total_class_entr
   
    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
   
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count
            entropy_class = - probability_class * np.log2(probability_class)
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
   
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy
       
    return calc_total_entropy(train_data, label, class_list) - feature_info

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None
   
    for feature in feature_list:
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
           
    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}
   
    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]
           
            if class_count == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature_name] != feature_value]
                assigned_to_node = True
        if not assigned_to_node:
            tree[feature_value] = "?"
           
    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(train_data, label, class_list)
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        next_root = None
       
        if prev_feature_value is not None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
       
        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, label, class_list)

def id3(train_data_m, label):
    train_data = train_data_m.copy()
    tree = {}
    class_list = train_data[label].unique()
    make_tree(tree, None, train_data, label, class_list)
    return tree

tree = id3(train_data_m, 'Buy_Computer')

def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{indent}|---> {key}")
            print_tree(value, indent + "  ")
    else:
        print(f"{indent}{tree}")

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None

def evaluate(tree, test_data_m, label):
    correct_predict = 0
    wrong_predict = 0
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]:
            correct_predict += 1
        else:
            wrong_predict += 1
    accuracy = correct_predict / (correct_predict + wrong_predict)
    return accuracy

def predict_class(tree, instance):
    return predict(tree, instance)

def get_user_input(features):
    user_input = {}
    for feature in features:
        user_input[feature] = input(f"Enter value for {feature}: ")
    return pd.Series(user_input)


test_data_m = pd.read_csv("Buy_Computer.csv")
test_data_m = test_data_m.drop(columns=['id'])   # Drop the 'id' column from test data

#accuracy = evaluate(tree, test_data_m, 'Buy_Computer')
#print(f"Accuracy: {accuracy}")

print("Decision Tree\n")
print_tree(tree)

features = ['age', 'income', 'student', 'credit_rating']  # Update with actual feature names
user_instance = get_user_input(features)

predicted_class = predict_class(tree, user_instance)   # Predicting class for the user input
print(f"\nTest Instance:\n{user_instance}")
print(f"Predicted Class: {predicted_class}")

# Optionally, display a message about actual class if it is known
def find_closest_instance(user_instance, dataset, features):
    """ Finds the closest instance in the dataset to the user-provided instance """
    for index, row in dataset.iterrows():
        if all(row[feature] == user_instance[feature] for feature in features):
            return row
    return None

closest_instance = find_closest_instance(user_instance, test_data_m, features)
if closest_instance is not None:
    actual_class = closest_instance['Buy_Computer']
    print(f"Actual Class: {actual_class}")

    
'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy

# Load data
data = pd.read_csv("Buy_Computer.csv")
data = data.drop(columns=['id'])  # Drop the 'id' column

# Encode categorical features as numerical values
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Define features and target variable
X = data.drop(columns=['Buy_Computer'])
y = data['Buy_Computer']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to calculate entropy using scipy
def calculate_entropy(y):
    value_counts = pd.Series(y).value_counts()
    probabilities = value_counts / len(y)
    return entropy(probabilities, base=2)

# Function to calculate information gain for a feature
def information_gain(data, feature, target):
    # Total entropy of the target
    total_entropy = calculate_entropy(data[target])

    # Weighted entropy of each feature value
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / len(data)) * calculate_entropy(data[data[feature] == values[i]][target])
        for i in range(len(values))
    )
    
    # Information gain is the difference
    return total_entropy - weighted_entropy

# Select best feature based on information gain
def find_best_feature(data, features, target):
    info_gains = {feature: information_gain(data, feature, target) for feature in features}
    return max(info_gains, key=info_gains.get)

# Manually create and train Decision Tree
best_feature = find_best_feature(X_train.assign(Buy_Computer=y_train), X_train.columns, 'Buy_Computer')
print(f"Best feature selected based on information gain: {best_feature}")

# Train Decision Tree Classifier for visualization
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Visualization using Entropy')
plt.show()

# Define function to get user input and encode it
def get_user_input(features, label_encoders):
    user_input = {}
    for feature in features:
        value = input(f"Enter value for {feature}: ")
        if feature in label_encoders:
            user_input[feature] = label_encoders[feature].transform([value])[0]
        else:
            user_input[feature] = int(value)
    return pd.DataFrame([user_input], columns=features)

# Predict based on user input
features = X.columns.tolist()
user_instance = get_user_input(features, label_encoders)
predicted_class = clf.predict(user_instance)
class_names = ['No', 'Yes']
predicted_class_label = class_names[predicted_class[0]]

# Print the prediction result
print(f"\nTest Instance:\n{user_instance}")
print(f"Predicted Class: {predicted_class_label}")'''
