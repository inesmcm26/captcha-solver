from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.models import load_model


def CNN(input_shape = (20, 20, 1)):
    model = Sequential()

    # 1st convolutional layer with max pooling
    model.add(Conv2D(20, (5, 5), padding="same", input_shape = input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2nd convolutional layer with max pooling
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation = 'relu'))

    # Output layer with 32 nodes -> number of possible characters
    model.add(Dense(62, activation = 'softmax'))

    # Compile the model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

    return model

def transfer_model(base_model_path, out_len):
    # get the base model
    base_model = load_model(base_model_path, compile=False)

    # create the new model by excluding the last layer and freezing all the layers
    model = Sequential()
    for layer in base_model.layers[:-1]: # go through until last layer
        model.add(layer)
        layer.trainable = False

    # add the new output layer
    model.add(Dense(out_len, activation = 'softmax'))

    # compile the model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

    return model


