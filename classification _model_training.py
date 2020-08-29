import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


data = []
label = []
classes = 43
current_path = os.getcwd()

for i in range(classes):
    path = os.path.join(current_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path+"\\"+a)
            image = image.resize((32, 32))
            image = np.array(image)
            data.append(image)
            label.append(i)
        except:
            print("Error loading image")
data = np.array(data)
label = np.array(label)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
X_train = np.array(list(map(preprocessing,X_train)))
X_test = np.array(list(map(preprocessing,X_test)))
X_train = X_train.reshape(31367, 32, 32, 1)
X_test = X_test.reshape(7842, 32, 32, 1)
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(X_train)

#Transforming output vectors to binary class matrix
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#Defining the CNN model layers and compiling the model
def cnn_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    #model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    #model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(43, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

model = cnn_model()
#Training the model
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50),steps_per_epoch=2000 ,epochs=10, validation_data=(X_test, y_test))
model.save('classification.hs')





