# Required Libraries

# Data manipulation
import pandas as pd

# Neural Network
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# Plot
import matplotlib.pyplot as plt
# %matplotlib inline

# Assessment
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Reading the dataset Kaggle
train = pd.read_csv("../input/digit-recognizer/train.csv")

# Alternative: read from keras itself
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Analysing the dataset
print("Qtd elements train: {}". format(len(train)))
print(train.head())

# Dividing x_train and y_train
Y = train["label"]
X = train.drop(labels = ["label"],axis = 1)
print(X.head())

# On format numpy-array of images dimension 28 x 28
# x = X.values.reshape(-1,28,28,1)

# See the other dataset in Y:
print(Y)

# I don't understand !
# Let's go see to matplotlib
plt.imshow(X.values[100].reshape(28,28), cmap=plt.cm.binary)
plt.show()
print('Label: {}'.format(Y[100]))

# Transforming the image 2d in a numpy-array (imagem 28*28 = 784 pixels)
x = X.values.reshape(42000, 784)

# Normalizing for values between [0, 1]:
x = x.astype('float32')
x /= 255
print(x[0])

# Equalizing the output-format:
num_classes = 10

# Converting for a output-vector with 10 dimensions
# example: 8 => [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y = keras.utils.to_categorical(Y, num_classes)
print(y[0])

# Separating a part for training (90%) and another for validation (10%)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state=9)
print('Qtd trains: {}'.format(len(x_train)))
print('Qtd validations: {}'.format(len(x_val)))
