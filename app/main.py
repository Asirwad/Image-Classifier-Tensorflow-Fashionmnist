import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model_object = load_model('models/model.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load the predefined dataset (70k of 28px*28px images)
dataset = datasets.fashion_mnist
img_width, img_height = 28, 28

# pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()


def predict(model, image, correct_label):
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label_correct, guess):
    plt.rcParams['text.color'] = 'green'
    plt.rcParams['axes.labelcolor'] = 'red'
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label_correct)
    plt.xlabel("Guess : " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_image_number_choice():
    while True:
        number = input("Pick a number: ")
        if number.isdigit():
            number = int(number)
            if 0 <= number <= 1000:
                return int(number)
            else:
                print("Try again!")


num = get_image_number_choice()
image_ = test_images[num]
label = test_labels[num]
predict(model=model_object, image=image_, correct_label=label)
