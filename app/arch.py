from tensorflow.keras import Sequential, layers


def get_model():
    model = Sequential([
        # input is a 28x28 image then flatten the 28x28 into a single 784x1 layer
        layers.Flatten(input_shape=(28, 28))

    ])
get_model()