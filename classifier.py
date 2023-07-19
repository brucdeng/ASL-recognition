import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sys


# Load the trained model
model = tf.keras.models.load_model('ASL_classification_model.h5', compile=False)


#Define the class labels for sign language gestures
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    resized_image = cv2.resize(image, (64, 64))
    # Normalize pixel values
    normalized_image = resized_image.astype("float32") / 255.0
    # Expand the dimensions of the image to match the input shape of the model
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image


# Function to add text to the image
def add_text_to_image(image, text, position, font_path, font_size, font_color=(0,0,0), thickness=1):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=font_color, stroke_width=thickness, stroke_fill=font_color)
    return np.array(pil_image)


# Load and preprocess the input image from command line
input_image_path = sys.argv[1]
input_image = cv2.imread(input_image_path)
preprocessed_image = preprocess_image(input_image)
reshaped_image = np.squeeze(preprocessed_image, axis=0)


# Predict
predictions = model.predict(preprocessed_image)
predicted_class_index = np.argmax(predictions[0])
prediction_accuracy = predictions[0][predicted_class_index]


# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]


# Set font configurations and add the text
font_path = 'fonts/Roboto-Thin.ttf'
font_size = 14
font_color = (255, 255, 255)
text_position = (10, 5)
text = f"Predicted Gesture: {predicted_class_label}\nPrediction Accuracy: {prediction_accuracy:.6f}"
output_image = add_text_to_image(input_image, text, text_position, font_path, font_size, font_color)


# Resize the image using interpolation and save
resized_image = cv2.resize(output_image, (500,500), interpolation=cv2.INTER_CUBIC)
output_path = 'output/output.jpg'  # Path to save the output image
cv2.imwrite(output_path, resized_image)
print(predictions)
print(predicted_class_index)
print("Predicted Gesture:", predicted_class_label)
print("Prediction Accuracy:", prediction_accuracy)