import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the trained model
model = load_model('path_to_your_trained_model.h5')  # Replace with the path to your trained model file

# Define class labels (the same labels used during training)
class_labels = [
    "Corn_common_rust",
    "Corn_healthy",
    "Corn_Infected",
    "Corn_northern_leaf_blight",
    "Corn_gray_leaf_spots"
]

# Load and preprocess the input image
img_path = 'path_to_your_input_image.jpg'  # Replace with the path to your input image file
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image data

# Make predictions
predictions = model.predict(img_array)

# Interpret the predictions
predicted_class_index = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class label
print("Predicted Class Label:", predicted_class_label)
