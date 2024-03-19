# Assignment header
"""
Austin Maxwell
February 4, 2024
Spring 2024 - Data 51100 - Section 002
Programming Assignment #3
"""

# Importing necessary libraries
import numpy as np


# Initialize variables needed for algorithm
training_path = r'/Users/austinmaxwell/iris-training-data.csv'
testing_path = r'/Users/austinmaxwell/iris-testing-data.csv'

# Load and parse datasets into seperate NumPy ndarrays
train_attributes = np.loadtxt(training_path, delimiter=",", usecols=(0,1,2,3))
test_attributes = np.loadtxt(testing_path, delimiter=",", usecols=(0,1,2,3))
train_labels = np.loadtxt(training_path, delimiter=",", usecols=4, dtype='U')
test_labels = np.loadtxt(testing_path, delimiter=",", usecols=4, dtype='U')

# Reshaping attribute arrays so distances can be calculated
reshaped_train_attributes = train_attributes[np.newaxis, :, :]
reshaped_test_attributes = test_attributes[:, np.newaxis, :]

# Saving pairwise subtraction of arrays to meet PEP8 style: 79 character line
pairwise_subtraction = reshaped_test_attributes - reshaped_train_attributes

# Calculate distance between each test and training case
distances = np.sqrt(np.sum((pairwise_subtraction)**2, axis=-1))

# Find the minimum distance and save index
nearest_neighbors = np.argmin(distances, axis=1)

# Map index stored in nearest_neighbor to generate prediction label
prediction_labels = train_labels[nearest_neighbors]

# Calculate accuracy by comparing the predicted labels with the test labels
accuracy = np.average(prediction_labels == test_labels) * 100

# Print the test and predicted class labels
print("#, True, Predicted")
for i in range(len(test_labels)):
    print(f"{i+1},{test_labels[i]},{prediction_labels[i]}")
    
# Print the accuracy
print(f"Accuracy: {accuracy:.2f}%")
