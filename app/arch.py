from tensorflow.keras import Sequential, layers


def get_model(img_width, img_height):
    model = Sequential([
        # input is a 28x28 image then flatten the 28x28 into a single 784x1 layer
        layers.Flatten(input_shape=(img_width, img_height)),

        # hidden layer is 128 deep
        layers.Dense(units=128, activation='relu'),

        # output is 0-9 (depending on what piece of cloth it is). return maximum
        layers.Dense(units=10, activation='softmax')

    ])
    return model