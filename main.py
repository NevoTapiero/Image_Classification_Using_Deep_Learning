import tensorflow as tf
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.models import load_model

# Configure the GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Define constants
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 10

# 1. Data Preparation
train_dir = r'C:\Users\nevot\Desktop\train_dir'
validation_dir = r'C:\Users\nevot\Desktop\validate_dir'

# Generator for training data with data augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,  # Rescaling factor. Defaults to None.
    rotation_range=40,  # Degree range for random rotations
    width_shift_range=0.2,  # Range for random horizontal shifts
    height_shift_range=0.2,  # Range for random vertical shifts
    shear_range=0.2,  # Shear Intensity (Shear angle in counter-clockwise direction)
    zoom_range=0.2,  # Range for random zoom
    horizontal_flip=True,  # Randomly flip inputs horizontally
    fill_mode='nearest'  # Strategy for filling in newly created pixels
)

validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for validation data

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='sparse'
)

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='sparse'
)


# 2. CNN Model Building

# Define the model path
model_path = r'C:\Users\nevot\PycharmProjects\Image_Classification_Using_Deep_Learning\best_model'

# Check if a previous model exists
if os.path.exists(model_path):
    print("Loading the previous model...")
    model = load_model(model_path)
else:
    print("Creating a new model...")
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.5),  # Add dropout with a 50% drop rate, adjust as needed
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(len(train_data_gen.class_indices))  # Number of classes
    ])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
    # Change here: Specify the file path without .h5 extension for native Keras format
    ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
]


# 4. Train the Model
history = model.fit(
    train_data_gen,
    steps_per_epoch = train_data_gen.samples // BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = val_data_gen,
    validation_steps =  val_data_gen.samples // BATCH_SIZE,
    callbacks = callbacks
)

# 5. Evaluate the Model
# Here, you could evaluate your model with the test set or validation set

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_data_gen, steps=val_data_gen.samples // BATCH_SIZE)

print("Validation Loss: ", val_loss)
print("Validation Accuracy: ", val_accuracy)


#6. Make Predictions
#To predict a new image, you would load the image, preprocess it to fit the model input size, then use model.predict()

#Replace with the local path to the image you want to predict
img_path = r'C:\Users\nevot\Desktop\validate_dir\Corn_Infected\20200701_082157.jpg'

#Load the image and resize it to the expected size
img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

#Convert the image to a numpy array and scale the pixel values
img_array = image.img_to_array(img) / 255.0

#Add a batch dimension (model expects images in batches)
img_batch = np.expand_dims(img_array, axis=0)

#Predict the class of the image
predictions = model.predict(img_batch)

#The prediction will be an array of probabilities for each class
#Convert these probabilities to class labels
predicted_class = np.argmax(predictions, axis=1)

#If you have the class names as a list, you can convert this to the actual name
class_names = ['Corn_common_rust', 'Corn_healthy', 'Corn_Infected', 'Corn_northern_leaf-blight','Crorn_gray_leaf_spots']
print("Predicted Class: ", class_names[predicted_class[0]])
