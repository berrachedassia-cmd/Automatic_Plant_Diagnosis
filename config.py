import os  # Provides functions to interact with the operating system (paths, folders, etc.)

# ==========================================================
# PATH CONFIGURATION
# ==========================================================
# This section defines ALL important directories used in the project.
# Centralizing paths here makes the project easier to maintain and portable.

# Absolute path to the folder where this config.py file is located
# __file__ → current file
# abspath → converts to absolute path
# dirname → gets the parent directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the original dataset (real images)
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "Palm_Leaves_Dataset")

# Path to the synthetic dataset (generated images after augmentation)
DATASET2_PATH = os.path.join(BASE_DIR, "data", "synthetic")

# Directory where trained models will be stored
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Full path to the saved CNN model file
MODEL_FILE = os.path.join(MODEL_DIR, "plant_cnn.keras")

# File where training history (accuracy, loss, etc.) will be saved
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.pkl")


# ==========================================================
# DATASET PARAMETERS
# ==========================================================
# These parameters control how images are loaded and processed.

# All images will be resized to 128x128 pixels
# This ensures uniformity for the neural network
IMAGE_SIZE = (128, 128)

# Number of images processed at once (batch)
# Smaller batch → less memory usage but slower training
# Larger batch → faster but requires more RAM/VRAM
BATCH_SIZE = 16

# Percentage of data reserved for validation
# (used to evaluate model during training)
VALIDATION_SPLIT = 0.2

# Random seed ensures reproducibility
# Same seed → same dataset split every time
SEED = 42


# ==========================================================
# CNN TRAINING PARAMETERS
# ==========================================================

# Number of classes (types of diseases)
# Initially unknown → will be set dynamically after loading dataset
NUM_CLASSES = None  

# Shape of input images for CNN
# (height, width, channels)
# 3 channels = RGB images
INPUT_SHAPE = (128, 128, 3)

# Learning rate controls how fast the model learns
# Too high → unstable training
# Too low → very slow learning
LEARNING_RATE = 0.0003

# Number of full passes over the dataset
EPOCHS = 30

# Early stopping patience:
# If validation loss does not improve for 5 epochs → stop training
PATIENCE = 5