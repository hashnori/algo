import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sys
import shutil

# Define the path to your main dataset directory
dataset_dir = 'dataset'  # Update with the correct path

# Get the list of all image files in the subdirectories
all_images = []
for subdirectory in os.listdir(dataset_dir):
    subdirectory_path = os.path.join(dataset_dir, subdirectory)
    if os.path.isdir(subdirectory_path):
        images_in_subdirectory = [os.path.join(subdirectory, f) for f in os.listdir(subdirectory_path) if f.endswith('.jpg')]
        all_images.extend(images_in_subdirectory)

# Check if there are any images in the dataset
if not all_images:
    print("No images found in the dataset directory. Exiting.")
    sys.exit(1)

# Split the dataset into training and validation sets
train_images, validation_images = train_test_split(all_images, test_size=0.2, random_state=42)

# Create separate directories for training and validation datasets
train_dir = 'training/dataset'  # Update with the correct path
validation_dir = 'validation/dataset'  # Update with the correct path

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Move images to the respective directories
for img in train_images:
    source_path = os.path.join(dataset_dir, img)
    destination_path = os.path.join(train_dir, img)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy(source_path, destination_path)

for img in validation_images:
    source_path = os.path.join(dataset_dir, img)
    destination_path = os.path.join(validation_dir, img)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy(source_path, destination_path)

# Set image size and batch size
img_size = (224, 224)
batch_size = 32



# Create data generators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,  # Add rotation for more variation
    width_shift_range=0.2,  # Add horizontal shift
    height_shift_range=0.2,  # Add vertical shift
    brightness_range=[0.8, 1.2],  # Adjust brightness
    fill_mode='nearest'  # Fill mode for pixel values outside the input boundaries
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
)

# Create the base model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Create the custom model
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Calculate steps_per_epoch for training
steps_per_epoch_train = len(train_generator) // batch_size

# Calculate steps_per_epoch for validation
steps_per_epoch_validation = len(validation_generator) // batch_size
# Print the length of the training generator
print("Length of the training generator:", len(train_generator))

# Calculate steps_per_epoch for training
steps_per_epoch_train = len(train_generator)
print("Steps per epoch for training:", steps_per_epoch_train)

# Print the length of the validation generator
print("Length of the validation generator:", len(validation_generator))

# Calculate steps_per_epoch for validation
steps_per_epoch_validation = len(validation_generator)
print("Steps per epoch for validation:", steps_per_epoch_validation)


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=10,  # Adjust the number of epochs based on your needs
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)


# Save the trained model
model.save('product_classifier_model.keras')
