import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

%matplotlib inline
np.random.seed(17)

# Load images and create label vector
root_path = "‪C:\Users\HP\Desktop\gaussian_filtered_images"
images_path = root_path + ""
images = [] # images
Y = [] # label vector

dir_names = [dir_name for dir_name in os.listdir(images_path) if os.path.isdir(images_path + dir_name)]
for dir_name in dir_names:
    if dir_name == "No_DR": # Create a label
        y = 0 
    else:
        y = 1
    for file_name in os.listdir(images_path + dir_name):
        images.append(plt.imread(images_path + dir_name + "/" + file_name))
        Y.append(y)
    print("Finish reading " + dir_name + " condition images.")
    print("Total image read: " + str(len(images)))
print("End of the data")
Y = np.reshape(np.array(Y), (1, -1)) # Make Y to be a row vector

# Print some detail
print("Size of each image: " + str(images[0].shape))
print("Size of labels: " + str(Y.shape))
print("Number of DR (y = 1): " + str(np.sum(Y)))
print("Number of No_DR (y = 0): " + str(Y.shape[1] - np.sum(Y)))

# Randomly show images
n = 4 # show 4 images
fig, axes = plt.subplots(nrows = 1, ncols = n, figsize=(16, 16))
rand_idx = np.random.randint(0, len(images), n)
for i in range(n):
    axes[i].imshow(images[rand_idx[i]])
    axes[i].title.set_text("Image #" + str(rand_idx[i]) + "\n y label = " + str(Y[0][rand_idx[i]]))
    axes[i].axis("off")
fig.show()

# Function to convert RGB to grayscale
def rgb_to_gray(image):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return np.dot(image, rgb_weights)

gray_images = list(map(rgb_to_gray, images)) # Convert RGB images to grayscale images

# Print the image size
print("Size of each grayscale image: " + str(gray_images[0].shape))

# Show grayscale images
fig, axes = plt.subplots(nrows = 1, ncols = n, figsize=(16, 16))
for i in range(n):
    axes[i].imshow(gray_images[rand_idx[i]], cmap='gray')
    axes[i].title.set_text("Image #" + str(rand_idx[i]) + "\n y label = " + str(Y[0][rand_idx[i]]))
    axes[i].axis("off")
fig.show()

# Prepare dataset matrix X
X = np.array(gray_images).reshape(len(gray_images), -1).T # flatten and reshape
print("Size of data set (n_x, m): " + str(X.shape))
print("Size of label (1, m): " + str(Y.shape))

# Initialize w and b function
def initialize_parameters(n_x):
    w = np.zeros((n_x, 1))
    b = 0
    return w, b

# Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Plot of sigmoid function
z = np.linspace(-4, 4, 101)
plt.figure(figsize = (5, 3))
_ = plt.plot(z, sigmoid(z))
plt.title("Sigmoid(z)")
plt.xlabel("z")
plt.grid()
plt.show()

# Forward and backward propagation
def propagate(X, Y, w, b):
    m = Y.shape[1]
    grads = {}
    A = sigmoid(w.T @ X + b) # forward propagation
    cost = -1/m * (np.sum (Y * np.log(A)) + np.sum((1 - Y) * np.log(1 - A))) # cost function
    
    # backward propagation
    grads['dw'] = 1/m * X @ (A - Y).T
    grads['db'] = 1/m * np.sum(A - Y, axis = 1, keepdims = True)
    
    return A, grads, cost   

# Logistic regression function
def logistic_regression(X, Y, learning_rate = 0.0006, num_iter = 200, print_cost = True):
    w, b = initialize_parameters(X.shape[0]) # initailize the parameters
    costs = []
    
    # logistic regression
    for i in range(num_iter):
        A, grads, cost = propagate(X, Y, w, b)
        if print_cost and i % 20 == 0:
            print("Iteration #" + str(i) + "\tCost value = " + str(cost))
        costs.append(cost)
        w -= learning_rate * grads['dw']
        b -= learning_rate * grads['db']
    
    # compute the cost of the final parameter
    A, grads, cost = propagate(X, Y, w, b)
    print("Final cost value = " + str(cost))
    costs.append(cost)
    
    return w, b, costs

# Split the data to train/test set
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size = 0.25, random_state = 5)
X_train, X_test, Y_train, Y_test = X_train.T, X_test.T, Y_train.T, Y_test.T

# Print some detail
m = Y.shape[1]
m_train, m_test = Y_train.shape[1], Y_test.shape[1]
y1_train, y1_test = np.sum(Y_train), np.sum(Y_test)
print("number of train samples: " + str(m_train) + "(" + "{0:.2f}".format(m_train/m*100) + "%)")
print("number of DR cases in train samples: " + str(y1_train) + "(" + "{0:.2f}".format(y1_train/m_train*100) + "%)")
print("number of test samples: " + str(m_test) + "(" + "{0:.2f}".format(m_test/m*100) + "%)")
print("number of DR cases in test samples: " + str(y1_test) + "(" + "{0:.2f}".format(y1_test/m_test*100) + "%)")

w, b = initialize_parameters(X.shape[0]) # initailize the parameters
w, b, costs = logistic_regression(X_train, Y_train, learning_rate = 0.0006, num_iter = 200)
_ = plt.plot(costs)
plt.xlabel("number of iterations")
plt.ylabel("Cost value")
plt.grid()
plt.show()

# Predict function
def predict(X, Y, w, b):
    A, _, _ = propagate(X, Y, w, b)
    A[A >= 0.5] = 1
    A[A < 0.5] = 0
    diff = np.abs(A - Y)
    acc = 1 - np.sum(diff)/diff.shape[1]
    return A, diff, acc

yhat_train, _, acc_train = predict(X_train, Y_train, w, b) # Accuracy on train set
print("Accuracy on train set: " + "{0:.2f}".format(acc_train*100) + "%")

yhat_test, _, acc_test = predict(X_test, Y_test, w, b) # Accuracy on test set
print("Accuracy on test set: " + "{0:.2f}".format(acc_test*100) + "%")