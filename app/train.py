import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, optimizers
import matplotlib.pyplot as plt

from app.arch import get_model

# load the predefined dataset (70k of 28px*28px images)
dataset = datasets.fashion_mnist
img_width, img_height = 28, 28

# pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

model = get_model(img_width, img_height)
model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=20)
model.save('models/model.h5')
