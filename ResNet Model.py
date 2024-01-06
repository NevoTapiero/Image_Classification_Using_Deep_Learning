import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping

# Input layer
input_layer = Input(shape=(32, 32, 3))

# Convolutional layers
conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
pooling = MaxPooling2D(pool_size=(2, 2))(conv2)

# Flatten layer
flatten = Flatten()(pooling)

# The following dense layer is the previous_layer
previous_layer = Dense(128, activation='relu')(flatten)

# Output layer with 5 units for classification
output_layer = Dense(5, activation='softmax')(previous_layer)
model = Model(inputs=input_layer, outputs=output_layer)

# Define your class labels
class_labels = [
    "Corn_common_rust",
    "Corn_healthy",
    "Corn_Infected",
    "Corn_northern_leaf_blight",
    "Corn_gray_leaf_spots"
]

# Define the number of classes
num_classes = len(class_labels)

# Define a mapping from class label to class index
class_to_index = {class_label: i for i, class_label in enumerate(class_labels)}

# Example: If you have a list of labels, e.g., ["Corn_healthy", "Corn_common_rust", "Corn_gray_leaf_spots"]
labels = ["Corn_common_rust", "Corn_healthy", "Corn_Infected", "Corn_northern_leaf_blight", "Corn_gray_leaf_spots"]

# Convert the list of labels to one-hot encoded vectors
one_hot_labels = []
for label in labels:
    one_hot_label = np.zeros(num_classes)
    one_hot_label[class_to_index[label]] = 1
    one_hot_labels.append(one_hot_label)

# Print the one-hot encoded labels
for label in one_hot_labels:
    print(label)

# Setting Training Hyperparameters
batch_size = 32  # Change according to your system
epochs = 200
data_augmentation = True
num_classes = 5  # Change according to your problem

# Assuming you have 32x32 RGB images, change accordingly
input_shape = (32, 32, 3)

# Define paths
train_dir = r'C:\Users\nevot\Desktop\train_dir'
test_dir = r'C:\Users\nevot\Desktop\validate_dir'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # Add data augmentation configurations here if needed
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define a learning rate schedule function
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr




saved_model_directory = r"C:\Users\nevot\PycharmProjects\Image_Classification_Using_Deep_Learning\saved_models"

# Compile the model with the desired optimizer and loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'mydataset_model.{epoch:03d}.h5'# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'mydataset_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Initialize checkpoint variable
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True)

# Check if the saved model exists in the directory
if os.path.exists(filepath):
    # Load the SavedModel
    model = load_model(filepath)

    # Compile the model with the desired optimizer and loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Loaded pre-trained model.")

else:
    # If the saved model doesn't exist, create and train a new model
    print("No pre-trained model found. Creating and training a new model.")

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1),
    cooldown=0,
    patience=5,
    min_lr=0.5e-6
)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping]


if early_stopping.stopped_epoch > 0:
    print(f"Early stopping at epoch {early_stopping.stopped_epoch + 1} due to no improvement in validation loss.")
else:
    print("Training completed without early stopping.")


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=callbacks
    )
else:
    print('Using real-time data augmentation.')
    # Set up data augmentation configuration here if needed
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=callbacks
    )


# Evaluate the model
scores = model.evaluate(test_generator)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
