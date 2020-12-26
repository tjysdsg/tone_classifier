from PIL import Image
from utils import preprocess_input
import numpy as np
import os
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D,Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model


def get_data():
    content = []
    label = []
    for i in range(5):
        datadir = os.path.join('feats', f'{i}')
        for f in os.scandir(datadir):
            if f.is_file() and '.jpg' in f.name:
                content.append(f.path)
                label.append(i)
    label = to_categorical(label, num_classes=5)
    ret = list(zip(content, label))
    return ret


class ImageLoader(Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.n = math.ceil(len(self.data) / self.batch_size)

        self.labels = np.asarray([d[1] for d in self.data], dtype='float32')
        self.files = [d[0] for d in self.data]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.asarray([
            np.asarray(Image.open(f), dtype='float32') for f in files
        ])
        x = tf.image.rgb_to_grayscale(x)
        return x, y

    def get_labels(self):
        return self.labels


def create_model(width, height, channels, activation):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(3, 3), padding='same', input_shape=(width, height, channels)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())  
    model.add(Dense(1024))  
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(1024))  
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Dense(5))  
    model.add(Activation('softmax'))  
    return model


def train_model(train_loader, val_loader, width, height, channels, lr, activation, epochs):
    model = create_model(width, height, channels, activation)
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print("====================== training ==========================")
    checkpoint_callback = ModelCheckpoint(filepath="ToneNet.hdf5", verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    model.fit(
        train_loader,
        epochs=epochs,
        validation_data=val_loader,
        verbose=1,
        workers=2,
        shuffle=False,
        use_multiprocessing=True,
        callbacks=[checkpoint_callback, earlystopping],
    )


def test_model(model_path, test_loader):
    print("====================== Testing ===========================")
    model = load_model(model_path)

    y = test_loader.get_labels()
    y = np.argmax(y, axis=1)
    y_pred = model.predict(test_loader)
    y_pred = np.argmax(y_pred, axis=1)

    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    accuracy = metrics.accuracy_score(y, y_pred)

    precision = metrics.precision_score(y, y_pred, average='macro')
    recall = metrics.recall_score(y, y_pred, average='macro')
    f1_score = 2 * recall * precision / (recall + precision)

    print(confusion_matrix)
    print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f1_score)


def main():
    width = 225
    height = 225
    channels = 1
    lr = 0.001
    activation = 'relu'
    epochs = 50
    batch_size = 128

    data = get_data()
    train, test = train_test_split(data, test_size=0.3, shuffle=True)
    train, val = train_test_split(train, test_size=0.1, shuffle=True)

    train_model(ImageLoader(train, batch_size), ImageLoader(val, batch_size), width, height, channels, lr, activation, epochs)
    test_model('ToneNet.hdf5', ImageLoader(test, batch_size))


if __name__ == "__main__":
    main()