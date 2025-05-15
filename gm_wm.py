import os
import numpy as np
import nibabel as nib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

files_disease = "E:\\pd_48\\c1\\"
files_healthy = "E:\\HC_FEB2023_SPM\\HC_FINAL_APRIL\\c1\\48\\"

start_time = time.time()

image_width = 30
image_height = 30
image_depth = 121

disease_images = []
healthy_images = []
labels = []

for i in range(1, 135):
    filename = os.path.join(files_disease, f"smwc1{i}.nii")
    if os.path.exists(filename):
        img = nib.load(filename)
        data = img.get_fdata()
        resized_data = cv2.resize(data, (image_width, image_height))
        disease_images.append(resized_data)
        labels.append(0)

for i in range(1, 135):
    filename = os.path.join(files_healthy, f"smwc1{i}.nii")
    if os.path.exists(filename):
        img = nib.load(filename)
        data = img.get_fdata()
        resized_data = cv2.resize(data, (image_width, image_height))
        healthy_images.append(resized_data)
        labels.append(1)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for data loading and resizing: {elapsed_time:.2f} seconds")

X = np.array(disease_images + healthy_images)
y = np.array(labels)

X_flattened = X.reshape(X.shape[0], -1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flattened)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.10, random_state=42, stratify=y)

X_train_cnn = X_train.reshape(X_train.shape[0], image_height, image_width, image_depth, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], image_height, image_width, image_depth, 1)
y_train_cnn = to_categorical(y_train, num_classes=2)
y_test_cnn = to_categorical(y_test, num_classes=2)


def plot_roc_curve(y_true, y_pred_proba, classifier_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {classifier_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{classifier_name}_ROC.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_confusion_matrix_func(y_true, y_pred, classifier_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['PD', 'HC'], yticklabels=['PD', 'HC'])
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{classifier_name}_CM.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity

classifiers = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
    "SVM-Linear": SVC(kernel='linear', probability=True, random_state=42),
    "SVM-RBF": SVC(kernel='rbf', probability=True, random_state=42),
    "SVM-Poly": SVC(kernel='poly', probability=True, random_state=42),
    "SVM-Sigmoid": SVC(kernel='sigmoid', probability=True, random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "SGD Classifier": SGDClassifier(loss='hinge', random_state=42, max_iter=1000, tol=1e-3),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training {name} completed in {end_time - start_time:.2f} seconds.")

    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
    else: # RidgeClassifier does not have predict_proba, use decision_function
        y_pred_proba = clf.decision_function(X_test)
        if len(y_pred_proba.shape) == 1: # Ensure it's 2D for ROC if needed, or scale for probability-like
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())


    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    sensitivity, specificity = plot_confusion_matrix_func(y_test, y_pred, name)

    print(f"Results for {name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {roc_auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print("-" * 30)

    results[name] = {'accuracy': accuracy, 'auc': roc_auc, 'f1': f1, 'sensitivity': sensitivity, 'specificity': specificity}
    plot_roc_curve(y_test, y_pred_proba, name)


print("\nTraining 3D CNN...")
start_time_cnn_total = time.time()

cnn_model = Sequential([
    Conv3D(16, kernel_size=(3, 3, 3), activation='relu', input_shape=(image_height, image_width, image_depth, 1), padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    BatchNormalization(),

    Conv3D(8, kernel_size=(3, 3, 3), activation='tanh', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    BatchNormalization(),

    Conv3D(4, kernel_size=(3, 3, 3), activation='tanh', padding='same'),
    BatchNormalization(),

    Flatten(),

    Dense(32, activation='tanh'),
    Dropout(0.5),
    Dense(8, activation='tanh'),
    Dropout(0.3),
    Dense(2, activation='sigmoid')
])

cnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

start_time_cnn_fit = time.time()
history = cnn_model.fit(
    X_train_cnn, y_train_cnn,
    validation_split=0.2,
    epochs=12,
    batch_size=4,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
end_time_cnn_fit = time.time()
print(f"CNN training completed in {end_time_cnn_fit - start_time_cnn_fit:.2f} seconds.")

loss, accuracy_cnn = cnn_model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
y_pred_proba_cnn = cnn_model.predict(X_test_cnn)[:, 1]
y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)

roc_auc_cnn = roc_auc_score(y_test, y_pred_proba_cnn)
f1_cnn = f1_score(y_test, y_pred_cnn)
sensitivity_cnn, specificity_cnn = plot_confusion_matrix_func(y_test, y_pred_cnn, "3D CNN")

print("\nResults for 3D CNN:")
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy_cnn:.4f}")
print(f"  AUC: {roc_auc_cnn:.4f}")
print(f"  F1 Score: {f1_cnn:.4f}")
print(f"  Sensitivity (Recall): {sensitivity_cnn:.4f}")
print(f"  Specificity: {specificity_cnn:.4f}")

results["3D CNN"] = {'accuracy': accuracy_cnn, 'auc': roc_auc_cnn, 'f1': f1_cnn, 'sensitivity': sensitivity_cnn, 'specificity': specificity_cnn}
plot_roc_curve(y_test, y_pred_proba_cnn, "3D CNN")

end_time_cnn_total = time.time()
print(f"Total CNN processing time (training + evaluation): {end_time_cnn_total - start_time_cnn_total:.2f} seconds.")

epochs_range = range(1, len(history.history['accuracy']) + 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('CNN_acc_history.png', dpi=600, bbox_inches='tight')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('CNN_loss_history.png', dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
plt.close()

print("\nSummary of All Classifier Results:")
for name, metrics_dict in results.items():
    print(f"\n{name}:")
    for metric_name, value in metrics_dict.items():
        print(f"  {metric_name.capitalize()}: {value:.4f}")