from tensorflow.keras import datasets
from tensorflow.keras.models import load_model

model = load_model('models/model.h5')

# load the predefined dataset (70k of 28px*28px images)
dataset = datasets.fashion_mnist
img_width, img_height = 28, 28

# pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

i = 0
num_success = 0
for image, actual_label in zip(test_images, test_labels):
    label_predicted = model.predict(image.reshape(1, img_width, img_height, 1))
    label_predicted_argmax = label_predicted.argmax()
    if label_predicted_argmax == actual_label:
        num_success += 1
        print(f"Image {i} , actual label: {actual_label}, prediction: {label_predicted_argmax}  ,Success")
    else:
        print(f"Image {i} , actual label: {actual_label}, prediction: {label_predicted_argmax}  ,Unsuccessful")
    i += 1
    if i == 100:
        break
print(f"Success rate = {num_success/100}")
