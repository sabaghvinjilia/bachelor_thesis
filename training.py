from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os


INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = "dataset"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = preprocess_input(img_to_array(load_img(img_path, target_size=(224, 224))))
        data.append(image)
        labels.append(category)

lb = LabelBinarizer()
labels = to_categorical(lb.fit_transform(labels))
data = np.array(data, dtype="float32")
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(data, labels,
                                                test_size=0.2,
                                                stratify=labels,
                                                random_state=70)

image_data_generator = ImageDataGenerator(
    rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, horizontal_flip=True,
    fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet",
                        include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

head = baseModel.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

model = Model(inputs=baseModel.input, outputs=head)
for layer in baseModel.layers:
    layer.trainable = False

optimizer = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(
    image_data_generator.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Evaluate and save
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
model.save("mask.h5")
