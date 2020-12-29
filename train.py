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


def create_model(width: int, height: int, channels: int, activation):
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

    model.add(Dense(4))  
    model.add(Activation('softmax'))  
    return model


def get_data():
    files = []
    label = []
    for i in range(4):
        with open(os.path.join('feats', f'{i}.list')) as f:
            for line in f:
                files.append(line.replace('\n', ''))
                label.append(i)
    label = to_categorical(label, num_classes=4)
    return list(zip(files, label))


def parse_data(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=1)
    # convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # images are already 225x225
    # image = tf.image.resize_images(image, [225, 225]) 

    return image, label


def create_data_pipeline(data, batch_size, n_workers=4, shuffle=True):
    filenames, labels = zip(*data)
    dataset = tf.data.Dataset.from_tensor_slices((list(filenames), list(labels)))
    if shuffle:
        dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_data, num_parallel_calls=n_workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def train_model(train_loader, val_loader, width, height, channels, lr, activation, epochs):
    model = create_model(width, height, channels, activation)
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print("====================== training ==========================")
    checkpoint_callback = ModelCheckpoint(filepath="ToneNet.hdf5", verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', verbose=1, mode='min')
    model.fit(
        train_loader,
        epochs=epochs,
        validation_data=val_loader,
        verbose=1,
        shuffle=False,
        use_multiprocessing=True,
        callbacks=[checkpoint_callback, earlystopping],
    )


def test_model(model_path, test_loader):
    print("====================== Testing ===========================")
    model = load_model(model_path)

    y = np.asarray([i for i in test_loader.unbatch().map(lambda _, y: y, num_parallel_calls=4)])
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

    print('=========================== loading data ===========================')
    data = get_data()
    train, test = train_test_split(data, test_size=0.2, shuffle=True)
    train, val = train_test_split(train, test_size=0.1, shuffle=True)

    train_model(
        create_data_pipeline(train, batch_size),
        create_data_pipeline(val, batch_size),
        width, height, channels, lr, activation, epochs
    )
    test_model('ToneNet.hdf5', create_data_pipeline(test, batch_size, shuffle=False))


if __name__ == "__main__":
    main()