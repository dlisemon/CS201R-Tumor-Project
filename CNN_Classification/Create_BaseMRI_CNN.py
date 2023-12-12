import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import logging
import pickle


logging.basicConfig(encoding='utf-8', level=logging.INFO)

with open('dataset.pkl', 'rb') as file:
    images_dataset = pickle.load(file)
    
desired_shape = (256, 256, 1)

X, y = zip(*images_dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("X and y flattened")


# Reshape and normalize
X_train = np.array(X_train).reshape(-1, desired_shape[0], desired_shape[1], desired_shape[2])
X_test = np.array(X_test).reshape(-1, desired_shape[0], desired_shape[1], desired_shape[2])
y_train = np.array(y_train)
y_test = np.array(y_test)
logging.info("Reshaped")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(desired_shape[0], desired_shape[1], desired_shape[2])),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu', name='dense_layer'),
    Dense(1, activation='sigmoid')  
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

model.save('BaseMRI_CNN.keras')

#To load keras model use:
# model = keras.models.load_model('path/to/location.keras')