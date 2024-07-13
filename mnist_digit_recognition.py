"""Reading handwritten digits: MNIST with 1, 2 and 3 hidden-layer networks in Keras.

Code exported from mnist_digit_recognition.ipynb; see the notebook for outputs and charts.
"""

# %%
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Input

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# %%
# Set seed
np.random.seed(9)

# %%
# Download the data
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# %%
# Print the shapes of the datasets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# %%
# Each observation represents an image of size 28x28
image_size = X_train.shape[1]
input_size = image_size * image_size

# Flatten each observation to a vector of length 784
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Print the Flatten shapes 
print(f"Flatten shapes -> X_train: {X_train.shape}")
print(f"Flatten shapes -> X_test: {X_test.shape}")

# %%
# Number of classes for classification
num_classes = 10

# Convert class vectors to binary class matrices 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# %%
# Function to display an image from a flattened vector
def display_image_from_vector(observation, label):
    image = observation.reshape((28, 28))
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')

# Number of random images 
num_images = 16

# Select random indices from the training set
random_indices = np.random.choice(X_train.shape[0], num_images, replace=False)

# Create a figure to display the images
plt.figure(figsize=(8, 8))
for i, index in enumerate(random_indices):
    plt.subplot(4, 4, i + 1)
    display_image_from_vector(X_train[index], np.argmax(y_train[index]))
plt.show()

# %%
# Model 1 - Single Hidden Layer
model_1 = Sequential()

# Input Layer
model_1.add(Input(shape=(784,)))

# First Layer 
model_1.add(Dense(units=512, input_shape=(784,), activation='relu'))

# Output Layer
model_1.add(Dense(units=10, activation='softmax'))

# %%
# Setup the model
model_1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
model_1.summary()

# %%
# Train the model 
model_1.fit(
    X_train, y_train,  
    epochs=3,  
    batch_size=32  # Number of samples per gradient update
)

# Evaluate the model on the test data
test_loss_1, test_acc_1 = model_1.evaluate(X_test, y_test)

# Print the test accuracy 
print(f'Test accuracy: {test_acc_1:.5f}')

# %%
# Model 2 - Two Hidden Layers
model_2 = Sequential()

# Input Layer
model_2.add(Input(shape=(784,)))

# First Layer
model_2.add(Dense(units=512, activation='relu'))

# Second Layer (the added one)
model_2.add(Dense(units=256, activation='relu'))

# Output Layer 
model_2.add(Dense(units=10, activation='softmax'))

# %%
# Setup the new model
model_2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
model_2.summary()

# %%
# Train the new model 
model_2.fit(
    X_train, y_train,  
    epochs=3,  
    batch_size=32  # Number of samples per gradient update
)

# Evaluate the new model 
test_loss_2, test_acc_2 = model_2.evaluate(X_test, y_test)

# Print the test accuracy 
print(f'Test accuracy: {test_acc_2:.5f}')

# %%
# Single Hidden Layer Model
model_1_name = "Single Hidden Layer Model"
model_1_test_accuracy = test_acc_1

# Two Hidden Layers Model
model_2_name = "Two Hidden Layers Model"
model_2_test_accuracy = test_acc_2

# Print the summary of the results
print(f"{model_1_name}\nTest Accuracy: {model_1_test_accuracy:.5f}\n")
print(f"{model_2_name}\nTest Accuracy: {model_2_test_accuracy:.5f}\n")

# %%
# Model 3 - Three Hidden Layers 
model_3 = Sequential()

# Input Layer
model_3.add(Input(shape=(784,)))

# First Layer
model_3.add(Dense(units=512, activation='relu'))

# Second Layer
model_3.add(Dense(units=256, activation='relu'))

# Third Layer (the one we try to add now)
model_3.add(Dense(units=128, activation='relu'))

# Output Layer 
model_3.add(Dense(units=10, activation='softmax'))

# %%
# Setup the Last model
model_3.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
model_3.summary()

# %%
# Train the Last model 
model_3.fit(
    X_train, y_train,  
    epochs=3,  
    batch_size=32  # Number of samples per gradient update
)

# Evaluate the Last model 
test_loss_3, test_acc_3 = model_3.evaluate(X_test, y_test)

# Print the test accuracy 
print(f'Test accuracy: {test_acc_3:.5f}')

# %%
# Three Hidden Layers Model
model_3_name = "Three Hidden Layers Model"
model_3_test_accuracy = test_acc_3

# Print the summary of the results
print(f"Single Hidden Layer Model\nTest Accuracy: {model_1_test_accuracy:.5f}\n")
print(f"Two Hidden Layers Model\nTest Accuracy: {model_2_test_accuracy:.5f}\n")
print(f"Three Hidden Layers Model\nTest Accuracy: {model_3_test_accuracy:.5f}\n")

# %%
# Best model - Three Hidden Layers Model
y_pred = model_3.predict(X_test) 
y_pred_classes = np.argmax(y_pred, axis=1)  
y_true = np.argmax(y_test, axis=1)  

# Misclassified samples
errors = (y_pred_classes != y_true)  
y_pred_classes_errors = y_pred_classes[errors]  
y_pred_errors = y_pred[errors]  
y_true_errors = y_true[errors]  
X_test_errors = X_test[errors]  

# Probabilities of predicted 
y_pred_errors_probability = np.max(y_pred_errors, axis=1)  
true_probability_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))  
diff_errors_pred_true = y_pred_errors_probability - true_probability_errors  

# List of indices of sorted differences
sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)  
top_idx_diff_errors = sorted_idx_diff_errors[-10:]  

# Top misclassified images
num = len(top_idx_diff_errors)  
f, ax = plt.subplots(1, num, figsize=(30, 30))  

for i in range(num):
    idx = top_idx_diff_errors[i]  
    sample = X_test_errors[idx].reshape(28, 28)  
    y_t = y_true_errors[idx]  
    y_p = y_pred_classes_errors[idx]  
    ax[i].imshow(sample, cmap='gray')  
    ax[i].set_title(f"Pred: {y_p}\nTrue: {y_t}", fontsize=12)  
    ax[i].axis('off')  

plt.tight_layout()  
plt.show() 

# Number of misclassified images
print(f"Total misclassified images: {len(y_pred_classes_errors)}")
