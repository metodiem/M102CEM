import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def parseImage(path, size):
    image = tf.image.decode_jpeg(tf.read_file(path), channels = 1)
    resized_image = tf.image.resize_images(image,[size, size])
    return resized_image

model = tf.keras.models.load_model("test_model.model")
#img = cv2.imread("two.jpg")
#resized = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
kur = []
kur.append(parseImage("two.jpg", 28))

print(kur[0])
#print(123)
predict = model.predict(np.argmax(kur[0]))
print(predict)