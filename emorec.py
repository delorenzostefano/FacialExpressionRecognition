import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

import itertools

import PIL.Image

# defining the variables
num_classes = 7
epochs = 25
batch_size = 256


# D:\Users\Stefano\Desktop\fer2013.csv
with open(r"D:\Users\Stefano\Desktop\fer2013.csv") as file:
    content = file.readlines()

lines = np.array(content)

num_instances = lines.size
print("number of instances: ", num_instances)
print("instance length: ", len(lines[1].split(",")[1].split(" ")))

x_train, y_train, x_test, y_test, x_val, y_val = [], [], [], [], [], []

# adding the training and test images to them variables
for i in range(1, num_instances):
    try:
        emotion, img, usage = lines[i].split(",")

        val = img.split(" ")

        pixels = np.array(val, 'float32')

        emotion = to_categorical(emotion, num_classes)

        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
        elif 'PrivateTest' in usage:
            y_val.append(emotion)
            x_val.append(pixels)
    except:
        print("", end="")

# --------------------------------------------------------
# data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')
x_val = np.array(x_val, 'float32')
y_val = np.array(y_val, 'float32')

# normalization
x_train /= 255
x_test /= 255
x_val /= 255

# reshaping images as 48x48
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')
x_val = x_val.reshape(x_val.shape[0], 48, 48, 1)
x_val = x_val.astype('float32')

# printing of the number of samples
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_val.shape[0], 'validation samples')

# -----------------------------------------------------------------
# defining callback to prevent overfitting
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]

# -----------------------------------------------------------------
# construct CNN structure
model = Sequential()

# 1st layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'))
model.add(BatchNormalization())

# 2nd layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Dropout(0.25))

# 3rd layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Dropout(0.25))

# 4th layer
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Dropout(0.25))

# classification
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# ------------------------------

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# ------------------------------
# training the model
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)


model.fit(train_generator,
          steps_per_epoch=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          callbacks=callbacks)

# ------------------------------

# evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])


# ----------------------------------
# function for drawing bar chart for emotion predictions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.show()


#--------------------------------------------
# function to print images from the dataset with their prediction
def check_img_dataset(first, second):
    # make predictions
    predictions = model.predict(x_test)
    # printing the images of the set from 2 to 34 with the emotion
    index = 0
    for i in predictions:
        if first < index < second:

            test_img = np.array(x_test[index], 'float32')
            test_img = test_img.reshape([48, 48]);

            plt.gray()
            plt.imshow(test_img)
            plt.show()

            emotion_analysis(i)
            # print("----------------------------------------------")
        index = index + 1


# ------------------------------------------------------
# test the model with custom images (only face)
def check_img(img_path):
    img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x /= 255

    custom = model.predict(x)
    emotion_analysis(custom[0])

    x = np.array(x, 'float32')
    x = x.reshape([48, 48])

    plt.gray()
    plt.imshow(x)
    plt.show()


# printing the recall and precision-----------------------------------------
def get_metrics():
    y_pred = model.predict_classes(x_test, verbose=0)
    # get 1 D
    y_true = np.argmax(y_test, axis=1)

    target_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


# ---------------------------------------------------------------------
# print the confusion matrix
def conf_mat():
    y_pred = model.predict_classes(x_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    title = 'Confusion matrix'
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------
