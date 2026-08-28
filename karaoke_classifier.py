#---IMPORTS
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.model_selection import train_test_split

#---CREATE CLASSES
folders = []
avg_discs = []
directory = "/Users/brandon/final-spis-project/sight-singing-vocal-data" #change this to match with your own directory
keyword = "label"

def calculate_weighted_avg(discrepancy):
    perfects = []
    alrights = []
    mids = []
    for value in discrepancy:
        if value < 0.25:
            perfects.append(value)
        elif value < 5.0:
            alrights.append(value)
        else:
            mids.append(value)

    return (sum(perfects)+sum(alrights)+(sum(mids)/2))/len(discrepancy)
                

def find_avg_disc(filename):
    # Grabbing the first and third columns (indexes 0 and 2)
    column_1 = []
    column_3 = []
    discrepancy = []

    with open(filename, "r") as file:
        for line in file:
            # split() automatically handles spaces and tabs, while strip() removes whitespace at the ends
            parts = line.strip().split()
            
            # Ensure the line isn't empty and has enough columns
            if len(parts) > 3:
                column_1.append(parts[1])
                column_3.append(parts[3])
            
    for i in range(len(column_1)):
        discrepancy.append(abs(round((float(column_1[i]) - float(column_3[i])), 2)))

    return calculate_weighted_avg(discrepancy)


for root, _, filenames in os.walk(directory):
    for filename in filenames:
        if keyword in filename:
            text_filename = os.path.join(root, filename)
            folders.append(text_filename)
            avg_discs.append(find_avg_disc(text_filename))

print(avg_discs)

for index, discrepancy in enumerate(avg_discs):
    if discrepancy < 0.25:
        print(f"Discrepancy for {folders[index]}: {discrepancy} is talented!")
    elif discrepancy < 5.0:
        print(f"Discrepancy for {folders[index]}: {discrepancy} is okay!")
    else:
        print(f"Discrepancy for {folders[index]}: {discrepancy} needs more work!")


y = discrepancy



# split data into training, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\nTraining size: {len(x_train)}, Validation size: {len(x_val)}, Test size: {len(x_test)}")

# Normalize pixel values to be between 0 and 1
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

# reshape y_train, y_val, and y_test to appropriate shape
y_train = np.array(y_train).reshape((y_train.shape[0], ))
y_val = np.array(y_val).reshape((y_val.shape[0], ))
y_test = np.array(y_test).reshape((y_test.shape[0], ))

# prepare data for CNN training
x_train = x_train.reshape(x_train.shape[0], 1200, 400, 3)
x_val = x_val.reshape(x_val.shape[0], 1200, 400, 3)
x_test = x_test.reshape(x_test.shape[0], 1200, 400, 3)

# convert labels to one-hot encoding
y_train_oh = tf.keras.utils.to_categorical(y_train, 3)
y_val_oh = tf.keras.utils.to_categorical(y_val, 3)
y_test_oh = tf.keras.utils.to_categorical(y_test, 3)

# Define the CNN model -- Trial and error qty. layers to fine-tune
model = Sequential()
model.add(Input(shape=(1200, 400, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) # notice no MaxPool after this one
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train_oh, epochs=10, batch_size=32, validation_data=(x_val, y_val_oh))

# Evaluate the model
y_pred = model.predict(x_test)
y_pred = tf.argmax(y_pred, axis=1)

loss, acc = model.evaluate(x_test, y_test_oh)
print("Test accuracy:", acc)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
     