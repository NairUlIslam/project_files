import os
import numpy as np
import nibabel as nib
import skimage.transform as skTrans
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

def load_and_preprocess_data(directory, categories, img_size):
    data = []
    count = 0
    for i in categories:
        folder = os.path.join(directory, i)
        for img_file in os.listdir(folder):
            if img_file.endswith('.nii.gz'):
                img_path = os.path.join(folder, img_file)
                label = categories.index(i)
                try:
                    img_arr = nib.load(img_path).get_fdata()
                    img_arr_resized = skTrans.resize(img_arr, img_size, order=1, preserve_range=True)
                    data.append([img_arr_resized, label])
                    count += 1
                except Exception as e:
                    print(f"Error loading or processing file {img_path}: {e}")
    return data, count

def create_cnn_model(sample_shape, l2_reg_factor=0.05):
    model = Sequential()
    l2_regularizer = regularizers.L2(l2_reg_factor)

    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', input_shape=sample_shape, padding='valid', kernel_regularizer=l2_regularizer))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(256, kernel_size=(2, 2, 2), activation='relu', kernel_regularizer=l2_regularizer))
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(512, kernel_size=(2, 2, 2), activation='relu', kernel_regularizer=l2_regularizer))
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization(momentum=0.6))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2_regularizer))
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2_regularizer))
    return model

def main():
    # Ensure this path is correct for your environment
    directory = '/content/drive/MyDrive/identified_1/' 
    categories = ['hc', 'pd'] # hc: healthy control, pd: Parkinson's disease
    img_size = (45, 54, 45) # As per paper/notebook
    
    raw_data, file_count = load_and_preprocess_data(directory, categories, img_size)
    print(f"Loaded {file_count} image files.")
    
    if not raw_data:
        print("No data loaded. Please check the directory path and file availability.")
        return

    X_list = []
    y_list = []
    for features, label in raw_data:
        X_list.append(features)
        y_list.append(label)
    
    X = np.asarray(X_list)
    y = np.asarray(y_list)
    
    print(f"Data shapes: X={X.shape}, y={y.shape}")

    # Splitting data: 90% for training/validation (for k-fold), 10% for final testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    # Reshape for CNN input (add channel dimension)
    X_train_val = X_train_val.reshape((*X_train_val.shape, 1))
    X_test = X_test.reshape((*X_test.shape, 1))
    
    print(f"Train/Validation shapes: X_train_val={X_train_val.shape}, y_train_val={y_train_val.shape}")
    print(f"Test shapes: X_test={X_test.shape}, y_test={y_test.shape}")

    # Parameters from the paper
    batch_size = 9
    no_epochs = 13
    learning_rate = 0.0001
    sample_shape = (img_size[0], img_size[1], img_size[2], 1)
    num_folds = 10
    l2_reg_factor = 0.05 # As specified in Table 2 of the paper
    
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42) # Added random_state for reproducibility
    
    fold_no = 1
    histories = []
    accuracies_test_set = []
    sensitivities_test_set = []
    specificities_test_set = []

    for train_idx, val_idx in kfold.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        model = create_cnn_model(sample_shape, l2_reg_factor=l2_reg_factor)
        
        model.compile(loss=binary_crossentropy, # Changed to binary_crossentropy as per standard for binary classification
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
        
        print(f'Training for fold {fold_no} ...')
        
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=no_epochs,
                            verbose=1, 
                            validation_data=(X_val, y_val))
        histories.append(history)
        
        # Evaluate on the held-out test set for this fold's model
        y_pred_test_proba = model.predict(X_test, verbose=0)
        y_pred_test_classes = (y_pred_test_proba > 0.5).astype(int)

        loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=0)
        accuracies_test_set.append(accuracy_test)
        
        report_test = classification_report(y_test, y_pred_test_classes, output_dict=True, zero_division=0)
        
        # Sensitivity = Recall for class "1" (PD)
        # Specificity = Recall for class "0" (HC)
        sensitivity_test = report_test.get("1", {}).get("recall", 0) 
        specificity_test = report_test.get("0", {}).get("recall", 0)
        
        sensitivities_test_set.append(sensitivity_test)
        specificities_test_set.append(specificity_test)
        
        print(f"Fold {fold_no} - Test Accuracy: {accuracy_test:.4f}, Test Sensitivity: {sensitivity_test:.4f}, Test Specificity: {specificity_test:.4f}")
        
        fold_no += 1
        
    # Plotting average accuracy and loss (optional, based on last fold's history for simplicity or average if needed)
    if histories:
        # Example: Plot metrics from the last fold
        last_history = histories[-1]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(last_history.history['accuracy'], label='Train Accuracy (Last Fold)')
        plt.plot(last_history.history['val_accuracy'], label='Validation Accuracy (Last Fold)')
        plt.title('Model Accuracy (Last Fold)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(last_history.history['loss'], label='Train Loss (Last Fold)')
        plt.plot(last_history.history['val_loss'], label='Validation Loss (Last Fold)')
        plt.title('Model Loss (Last Fold)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.legend(loc='upper left')
        plt.show()
        
    avg_test_accuracy = np.mean(accuracies_test_set)
    avg_test_sensitivity = np.mean(sensitivities_test_set)
    avg_test_specificity = np.mean(specificities_test_set)
    
    print(f'\nAverage Test Accuracy over {num_folds} folds: {avg_test_accuracy:.4f}')
    print(f'Average Test Sensitivity over {num_folds} folds: {avg_test_sensitivity:.4f}')
    print(f'Average Test Specificity over {num_folds} folds: {avg_test_specificity:.4f}')

if __name__ == '__main__':
    main()
