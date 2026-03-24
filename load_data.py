import tensorflow as tf
from config import DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, SEED

def load_datasets():
    # ==========================================================
    # LOAD DATASET FROM DIRECTORY
    # ==========================================================
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    # Extract class names
    class_names = full_dataset.class_names
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Number of classes:", num_classes)

    # ==========================================================
    # SPLIT DATASET INTO TRAIN / VALIDATION / TEST
    # ==========================================================
    dataset_size = len(full_dataset)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)

    # take() → first N batches
    # skip() → skip N batches
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size).take(val_size)
    test_dataset = full_dataset.skip(train_size + val_size)

    # ==========================================================
    # DATA AUGMENTATION (TRAIN ONLY)
    # ==========================================================
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # ==========================================================
    # PREPROCESSING (MobileNetV2 REQUIREMENT)
    # ==========================================================
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    # Apply transformations
    train_dataset = train_dataset.map(lambda x, y: (preprocess(data_augmentation(x)), y))
    val_dataset = val_dataset.map(lambda x, y: (preprocess(x), y))
    test_dataset = test_dataset.map(lambda x, y: (preprocess(x), y))

    # ==========================================================
    # PERFORMANCE OPTIMIZATION
    # ==========================================================
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, class_names, num_classes