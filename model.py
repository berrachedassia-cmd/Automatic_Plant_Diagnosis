import tensorflow as tf
from keras import layers, models

def build_cnn_model(num_classes, input_shape=(128,128,3)):

    # ==========================================================
    # LOAD PRETRAINED MODEL (TRANSFER LEARNING)
    # ==========================================================
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,   # Remove original classifier
        weights='imagenet'   # Use pretrained weights
    )

    # Freeze base model → prevents weights from being updated
    base_model.trainable = False

    # ==========================================================
    # BUILD CUSTOM CLASSIFIER
    # ==========================================================
    model = models.Sequential([
        base_model,

        # Convert feature maps → vector
        layers.GlobalAveragePooling2D(),

        # Normalize activations
        layers.BatchNormalization(),

        # Fully connected layer
        layers.Dense(128, activation='relu'),

        # Regularization to prevent overfitting
        layers.Dropout(0.5),

        # Final classification layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model