import os
import cv2
import numpy as np
import pickle
from glob import glob
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import config

# ===============================
# LOAD SYNTHETIC DATASET (DATASET 2)
# ===============================
def load_synthetic_dataset(image_size=(64,64)):
    DATASET_PATH = config.DATASET2_PATH  

    X, y = [], []

    # Vérification du path
    if not os.path.exists(DATASET_PATH):
        raise ValueError(f"Dataset 2 not found at {DATASET_PATH}")

    class_folders = sorted(os.listdir(DATASET_PATH))

    for label, folder in enumerate(class_folders):
        folder_path = os.path.join(DATASET_PATH, folder)

        # Support jpg + png (robustesse)
        images = glob(os.path.join(folder_path, "*.jpg")) + \
                 glob(os.path.join(folder_path, "*.png"))

        for img_path in images:
            img = cv2.imread(img_path)

            # Vérifier image valide
            if img is None:
                continue

            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            X.append(img.flatten())
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)

    return X, y, class_folders


# ===============================
# LOAD DATASET 2
# ===============================
X_syn, y_syn, class_names_syn = load_synthetic_dataset()

print(f"Synthetic dataset shape: {X_syn.shape}")
print(f"Number of classes: {len(class_names_syn)}")
print(f"Classes: {class_names_syn}")

# ===============================
# DEFINE PIPELINE (SCALER + PCA + SVM)
# ===============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=150, random_state=config.SEED)),
    ("svm", SVC(kernel="rbf", probability=True, random_state=config.SEED))
])

# ===============================
# STRATIFIED K-FOLD CROSS-VALIDATION
# ===============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

print("\nRunning 5-Fold Stratified Cross Validation on synthetic dataset...")

cv_scores = cross_val_score(
    pipeline,
    X_syn,
    y_syn,
    cv=skf,
    scoring="accuracy",
    n_jobs=1
)

print(f"\nCross-Validation Accuracy Mean: {cv_scores.mean():.4f}")
print(f"Cross-Validation Accuracy Std: {cv_scores.std():.4f}")

# ===============================
# SAVE CV MEAN ACCURACY
# ===============================
models_dir = os.path.join(config.BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

ml_acc_path = os.path.join(models_dir, "ml_accuracy.pkl")

with open(ml_acc_path, "wb") as f:
    pickle.dump(cv_scores.mean(), f)

print(f"\nML accuracy (CV mean) saved at: {ml_acc_path}")
print("SVM cross-validation pipeline complete.")