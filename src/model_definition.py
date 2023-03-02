from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
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

    # Compile the model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

    return model

def complex_CNN(input_shape = (20, 20, 1)):
    model = Sequential()

    # 1st convolutional layer with max pooling -> 20 kernels
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2nd convolutional layer with max pooling -> 50 kernels
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2nd convolutional layer with max pooling -> 50 kernels
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    # Fully Connected network
    # Hidden layer with 500 nodes
    model.add(Dense(500, activation = 'relu'))

    # Output layer with 32 nodes -> number of possible characters
    model.add(Dense(62, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

    return model

def VGG(input_shape = (20, 20, 1)):
    model = Sequential()
    
    
    model.add(Conv2D(256, (5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Conv2D(384, (3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization()),

    model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    # 62 possible characters
    model.add(Dense(62, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

    return model

