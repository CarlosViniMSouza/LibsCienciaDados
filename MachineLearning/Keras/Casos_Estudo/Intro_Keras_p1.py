# Required Libraries

# Data manipulation
import pandas as pd
import numpy as np

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
import itertools

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

# Creating and Training the model

model = Sequential()

# Layer with 30 neurons
model.add(Dense(30, activation='relu', input_shape=(784,)))

# Dropout of 20%
model.add(Dropout(0.2))

# Layer with 20 neurons
model.add(Dense(20, activation='relu'))

# Dropout of 20%
model.add(Dropout(0.2))

# Layer final classification, with 1 neuron by output class. Softmax divide a probability in classes.
model.add(Dense(num_classes, activation='softmax'))

# A resume of situation
model.summary()

# Compilation the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Train with anyone dates
batch_size = 32
epochs = 30
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val))

# Let's see how the training went
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# Testing
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Testing a anyone data input
print(y_train[10])
print(model.predict(x_train[10].reshape((1,784))))
print(model.predict_classes(x_train[10].reshape((1,784))))

# Evaluating the Model

# Plot the confusion matrix. Set Normalize = True/False
def plot_confusion_matrix(cm, classes, normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# See any reports with sklearn

# Sorting the entire test base
y_pred = model.predict_classes(x_val)

# Going back to class format
y_test_c = np.argmax(y_val, axis=1)
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred)
plot_confusion_matrix(cm, target_names, normalize=False, title='Confusion Matrix')

# Classification Report
print('Classification Report: ')
print(classification_report(y_test_c, y_pred, target_names=target_names))