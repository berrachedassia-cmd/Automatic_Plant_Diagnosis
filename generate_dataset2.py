import os
import tensorflow as tf
from config import DATASET_PATH, DATASET2_PATH  

# All images will be resized to this size
IMAGE_SIZE = (128, 128)

# ==========================================================
# DATA AUGMENTATION PIPELINE
# ==========================================================
# This block creates random transformations to artificially increase dataset size.
# It helps prevent overfitting and improves generalization.

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip image horizontally
    tf.keras.layers.RandomRotation(0.3),       # Rotate image randomly (±30%)
    tf.keras.layers.RandomZoom(0.3),           # Zoom in/out randomly
    tf.keras.layers.RandomContrast(0.3),       # Adjust contrast
    tf.keras.layers.RandomBrightness(0.2),     # Adjust brightness
])

# ==========================================================
# LOAD ORIGINAL DATASET
# ==========================================================
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,      # Root folder containing class subfolders
    image_size=IMAGE_SIZE,
    batch_size=1,      # Process one image at a time
    shuffle=False      # Keep order stable (important for labeling)
)

# Retrieve class names (folder names)
class_names = dataset.class_names

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================
# For each class, create a corresponding folder in synthetic dataset
for class_name in class_names:
    os.makedirs(os.path.join(DATASET2_PATH, class_name), exist_ok=True)

# ==========================================================
# GENERATE SYNTHETIC IMAGES
# ==========================================================
count = 0  # Counter for naming images

for images, labels in dataset:
    # Apply augmentation to the image
    augmented = data_augmentation(images)

    # Extract label (convert tensor → numpy)
    label = labels.numpy()[0]

    # Convert label index to class name
    class_name = class_names[label]

    # Build file path for saving image
    save_path = os.path.join(
        DATASET2_PATH,
        class_name,
        f"img_{count}.jpg"
    )

    # Save augmented image
    tf.keras.utils.save_img(save_path, augmented[0])

    count += 1

print(f"Dataset 2 generated successfully at {DATASET2_PATH}!")