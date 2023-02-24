from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

def simple_CNN(input_shape = (20, 20, 1)):
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
    model.add(Dense(32, activation = 'softmax'))

    # Ask Keras to build the model with TensorFlow behind the scenes
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

    return model