from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

wordsList=[]
with open("words.txt", 'r') as file:
    wordsList = [line.lower().strip() for line in file.readlines()]
    
file_name = "nor_oneH.csv"

# Load your dataset
df = pd.read_csv(file_name)
# Assuming 'Label' is your target variable
X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Make sure get_vocab and insert_feature functions work with DataFrames correctly
def get_vocab(X_train):
    vocab = set()
    pattern = r"[^\w\s]"
    # Assuming the text is in the 4th column, adjust the index as necessary
    texts = X_train.iloc[:, 3].fillna("").astype(str)  # Handle NaN values and ensure string type
    for text in texts:
        cleaned_text = re.sub(pattern, " ", text)
        words = cleaned_text.lower().split()
        vocab.update(word for word in words if word in wordsList)
    return vocab


def insert_feature(df, vocab):
    # Extract and clean the text column, ensuring all entries are treated as strings
    texts = df.iloc[:, 3].fillna("").astype(str).apply(lambda x: set(re.sub(r"[^\w\s]", " ", x).lower().split()))
    features = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, words in enumerate(texts):
        for j, word in enumerate(vocab):
            if word in words:
                features[i, j] = 1.0
    return features


# Apply feature extraction
vocab = get_vocab(X_train)
print(len(vocab))
vocab_list = list(vocab)

# Open a file in write mode. If the file does not exist, it will be created.
with open('NN_vocab.txt', 'w') as f:
    # Write the vocabulary list in the desired format
    f.write('[' + ', '.join([f'"{word}"' for word in vocab_list]) + ']')

X_train_features = insert_feature(X_train, vocab)
X_test_features = insert_feature(X_test, vocab)

print(X_train_features.shape)

# Combine the original numeric data (excluding the text column) with the new features
X_train_combined = np.hstack([X_train.drop(X_train.columns[3], axis=1).values.astype(np.float64), X_train_features])
X_test_combined = np.hstack([X_test.drop(X_test.columns[3], axis=1).values.astype(np.float64), X_test_features])

print(X_train_combined.shape)
# Define and train the model
# Adjust hyperparameters as needed. Here's a starting point based on your custom model
mlp = MLPClassifier(hidden_layer_sizes=(150), max_iter=250, alpha=1e-4,
                    solver='sgd',
                    learning_rate_init=0.01)

mlp.fit(X_train_combined, y_train)

# classification_reports = []

# y_pred_fold = mlp.predict(X_test_combined)
# fold_accuracy = accuracy_score(y_test, y_pred_fold)

# New part: Generating and printing classification report for the current fold
# report = classification_report(y_test, y_pred_fold, output_dict=False)
# classification_reports.append(report)
# print(f"Classification Report for Current Fold:\n{report}")

# # New part: Computing confusion matrix for the current fold
# conf_matrix = confusion_matrix(y_test, y_pred_fold)
# fig, ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
#             xticklabels=["Class 1", "Class 2", "Class 3", "Class 4"],
#             yticklabels=["Class 1", "Class 2", "Class 3", "Class 4"])
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()

# # Make predictions and evaluate the model
y_pred = mlp.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
y_train_pred = mlp.predict(X_train_combined)

# Calculate and print the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

    # Extract weights and biases
weights_first_layer = mlp.coefs_[0]
biases_first_layer = mlp.intercepts_[0]

print(weights_first_layer.shape)
# For models with more than one layer, extract second layer weights and biases
if len(mlp.coefs_) > 1:
    weights_second_layer = mlp.coefs_[1]
    biases_second_layer = mlp.intercepts_[1]

# Save to text files
np.savetxt("./NN_weights_1st_layer.txt", weights_first_layer)
np.savetxt("./NN_biases_1st_layer.txt", biases_first_layer)

if len(mlp.coefs_) > 1:
    np.savetxt("./NN_weights_2nd_layer.txt", weights_second_layer)
    np.savetxt("./NN_biases_2nd_layer.txt", biases_second_layer)

cities = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']
final_predictions = [cities[int(i)] for i in y_test]
# Convert the list of city names to a string representation
final_predictions_str = str(final_predictions)

# Write the string representation of the list to a text file
with open('NN_label.txt', 'w') as f:
    f.write(final_predictions_str)