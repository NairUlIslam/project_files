import pandas as pd
import os
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

BASE_DATA_PATH = r'E:\\subcorticalmi'
HEALTHY_CSV_PATH = os.path.join(BASE_DATA_PATH, 'T1_HC_FULL_AUG2022_8_27_2022.csv')
HEALTHY_FILES_DIR = os.path.join(BASE_DATA_PATH, 'hc', 'output')
DISEASE_CSV_PATH = os.path.join(BASE_DATA_PATH, 'PD_Baseline_7_Feb_7_28_2023.csv')
DISEASE_FILES_DIR = os.path.join(BASE_DATA_PATH, 'pd_baseline_7_febb', 'amyg_pd', 'output')
NEW_IMAGE_SHAPE = (70, 70, 70)
RANDOM_SEED = 42
TEST_SIZE = 0.2

STRUCTURE_SUFFIXES_TO_PROCESS = [
    "_togetherLPutaRPutaallnonefirstseg_structure.nii.gz",
    "_togetherLThalRThalallnonefirstseg_structure.nii.gz",
    "_togetherBrStem-BrStemfirst_structure.nii.gz",
    "_togetherLAccuRAccuallfastfirstseg_structure.nii.gz",
    "_togetherLAmygRAmygallfastfirstseg_structure.nii.gz",
    "_togetherLCaudRCaudallfastfirstseg_structure.nii.gz",
    "_togetherLHippRHippallfastfirstseg_structure.nii.gz",
    "_togetherLPallRPallallnonefirstseg_structure.nii.gz"
]

def resample_image(img, new_shape):
    old_shape = img.shape
    factors = [new_dim / old_dim for old_dim, new_dim in zip(old_shape, new_shape)]
    return zoom(img, factors, order=1)

def load_subject_ids_for_gender(csv_path, gender):
    try:
        csv_data = pd.read_csv(csv_path)
        return csv_data[csv_data["Sex"] == gender]["Subject"].tolist()
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return []

def load_and_process_structure_data(
    healthy_csv, healthy_dir, disease_csv, disease_dir,
    structure_file_suffix, image_shape, target_gender
):
    print(f"\n--- Processing structure: {structure_file_suffix} for gender: {target_gender} ---")

    healthy_subject_ids = load_subject_ids_for_gender(healthy_csv, target_gender)
    disease_subject_ids = load_subject_ids_for_gender(disease_csv, target_gender)

    if not healthy_subject_ids or not disease_subject_ids:
        print("Could not load subject IDs for one or both groups. Skipping this structure.")
        return None, None

    all_image_data = []
    loaded_healthy_subjects = set()
    loaded_disease_subjects = set()

    print(f"Loading healthy data from: {healthy_dir}")
    for root, _, files in os.walk(healthy_dir):
        for file in files:
            if file.endswith(structure_file_suffix):
                try:
                    subject_id_str = file.split('_')[1]
                    subject_id = int(subject_id_str)
                    if subject_id in healthy_subject_ids:
                        file_path = os.path.join(root, file)
                        img = nib.load(file_path).get_fdata()
                        rescaled_img = resample_image(img, image_shape)
                        all_image_data.append((rescaled_img, 'healthy'))
                        loaded_healthy_subjects.add(subject_id)
                except Exception as e:
                    print(f"Warning: Could not process healthy file {file}: {e}")
    print(f"Found {len([item for item in all_image_data if item[1] == 'healthy'])} healthy images from {len(loaded_healthy_subjects)} unique subjects.")

    print(f"Loading disease data from: {disease_dir}")
    current_disease_image_count_start = len(all_image_data)
    for root, _, files in os.walk(disease_dir):
        for file in files:
            if file.endswith(structure_file_suffix):
                try:
                    subject_id_str = file.split('_')[1]
                    subject_id = int(subject_id_str)
                    if subject_id in disease_subject_ids:
                        file_path = os.path.join(root, file)
                        img = nib.load(file_path).get_fdata()
                        rescaled_img = resample_image(img, image_shape)
                        all_image_data.append((rescaled_img, 'disease'))
                        loaded_disease_subjects.add(subject_id)
                except Exception as e:
                    print(f"Warning: Could not process disease file {file}: {e}")
    disease_images_loaded_this_run = len(all_image_data) - current_disease_image_count_start
    print(f"Found {disease_images_loaded_this_run} disease images from {len(loaded_disease_subjects)} unique subjects.")


    if not all_image_data:
        print("No images loaded. Skipping.")
        return None, None

    healthy_images_data = [item for item in all_image_data if item[1] == 'healthy']
    disease_images_data = [item for item in all_image_data if item[1] == 'disease']

    if not healthy_images_data or not disease_images_data:
        print("One of the groups has no images after filtering. Skipping.")
        return None, None

    min_files = min(len(healthy_images_data), len(disease_images_data))
    print(f"Balancing classes: Using {min_files} samples per class.")
    if min_files == 0:
        print("Zero files in the minimum set after attempting to load. Skipping.")
        return None, None

    random.seed(RANDOM_SEED)
    healthy_images_sampled = random.sample(healthy_images_data, min_files)
    disease_images_sampled = random.sample(disease_images_data, min_files)

    images, labels = zip(*(healthy_images_sampled + disease_images_sampled))
    return np.array(images), np.array(labels)


def get_model_definitions():
    standard_models = {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=RANDOM_SEED),
        'Linear SVM': SVC(kernel='linear', random_state=RANDOM_SEED),
        'KNN': KNeighborsClassifier(),
        'Gaussian Naive Bayes': GaussianNB(),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED),
        'DecisionTree': DecisionTreeClassifier(max_depth=15, random_state=RANDOM_SEED),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=RANDOM_SEED, algorithm='SAMME'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED)
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': RANDOM_SEED,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    xgb_model_def = xgb.XGBClassifier(**xgb_params)

    return standard_models, xgb_model_def

def evaluate_model_metrics(model, X_test_data, y_test_data, is_xgboost=False, label_encoder_instance=None):
    if is_xgboost and label_encoder_instance:
        y_pred = model.predict(X_test_data)
        y_test_transformed = label_encoder_instance.transform(y_test_data)
    else:
        y_pred = model.predict(X_test_data)
        y_test_transformed = y_test_data

    accuracy = model.score(X_test_data, y_test_transformed if is_xgboost else y_test_data)

    if is_xgboost:
        try:
            tn, fp, fn, tp = confusion_matrix(y_test_transformed, y_pred).ravel()
        except ValueError as e:
            print(f"CM Error (XGB): {e}. Unique y_test_transformed: {np.unique(y_test_transformed)}, Unique y_pred: {np.unique(y_pred)}")
            return accuracy, np.nan, np.nan
    else:
        try:
            tn, fp, fn, tp = confusion_matrix(y_test_data, y_pred, labels=['healthy', 'disease']).ravel()
        except ValueError as e:
            print(f"CM Error: {e}. Unique y_test_data: {np.unique(y_test_data)}, Unique y_pred: {np.unique(y_pred)}")
            return accuracy, np.nan, np.nan

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    return accuracy, specificity, sensitivity

def train_evaluate_all_models(X_train_data, y_train_data, X_test_data, y_test_data, std_models, xgb_model_instance):
    print("\n--- Training and Evaluating Models ---")

    for model_name, model in std_models.items():
        try:
            print(f"Training {model_name}...")
            model.fit(X_train_data, y_train_data)
            accuracy, specificity, sensitivity = evaluate_model_metrics(model, X_test_data, y_test_data)
            print(f"{model_name}: Accuracy={accuracy:.4f}, Specificity={specificity:.4f}, Sensitivity={sensitivity:.4f}")
        except Exception as e:
            print(f"Error training/evaluating {model_name}: {e}")

    if xgb_model_instance:
        print("Training XGBoost...")
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_data)
        y_test_encoded = le.transform(y_test_data)
        try:
            xgb_model_instance.fit(X_train_data, y_train_encoded)
            accuracy, specificity, sensitivity = evaluate_model_metrics(xgb_model_instance, X_test_data, y_test_encoded, is_xgboost=True, label_encoder_instance=le)
            print(f"XGBoost: Accuracy={accuracy:.4f}, Specificity={specificity:.4f}, Sensitivity={sensitivity:.4f}")
            print(f"XGBoost LabelEncoder classes: {le.classes_} (0: {le.classes_[0]}, 1: {le.classes_[1]})")
        except Exception as e:
            print(f"Error training/evaluating XGBoost: {e}")

if __name__ == "__main__":
    target_gender = input("Enter the gender for analysis (M or F): ").upper()
    if target_gender not in ['M', 'F']:
        print("Invalid gender specified. Exiting.")
        exit()

    for structure_suffix in STRUCTURE_SUFFIXES_TO_PROCESS:
        images_data, labels_data = load_and_process_structure_data(
            HEALTHY_CSV_PATH, HEALTHY_FILES_DIR,
            DISEASE_CSV_PATH, DISEASE_FILES_DIR,
            structure_suffix, NEW_IMAGE_SHAPE, target_gender
        )

        if images_data is None or labels_data is None or len(images_data) == 0:
            print(f"No data loaded for structure {structure_suffix}. Skipping.")
            continue

        print(f"Total images loaded for {structure_suffix}: {len(images_data)}")
        print(f"Shape of images_data: {images_data.shape}, labels_data: {labels_data.shape}")

        num_samples = images_data.shape[0]
        images_flattened = images_data.reshape(num_samples, -1)
        print(f"Shape of flattened images: {images_flattened.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            images_flattened, labels_data, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels_data
        )
        print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
        print(f"Unique labels in y_train: {np.unique(y_train, return_counts=True)}")
        print(f"Unique labels in y_test: {np.unique(y_test, return_counts=True)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        standard_ml_models, xgb_ml_model = get_model_definitions()

        train_evaluate_all_models(
            X_train_scaled, y_train, X_test_scaled, y_test,
            standard_ml_models, xgb_ml_model
        )
