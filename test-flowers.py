import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load the model
model = load_model('best_model.keras')

# Load the test data
data = pd.read_csv('test_nowy.csv')
X_test = data['filepaths'].values
y_test = data['labels'].values

# Function to load and preprocess images
def preprocess_npy(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    image = np.load(image_path)
    image = np.resize(image, (128, 128, 3))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# Preprocess all .npy files
X_test_processed = []
valid_image_paths = []
for npy_path in X_test:
    # print(f"Processing .npy file: {npy_path}")  # Debugging print statement
    try:
        processed_image = preprocess_npy(npy_path)
        X_test_processed.append(processed_image)
        valid_image_paths.append(npy_path)
    except FileNotFoundError as e:
        print(e)
        continue

# Check if the list is empty
if not X_test_processed:
    raise ValueError("No valid images found. Ensure the file paths are correct.")

X_test_processed = np.vstack(X_test_processed)

# Update y_test to only include valid image paths
y_test_filtered = [label for img_path, label in zip(X_test, y_test) if img_path in valid_image_paths]

# Predict on the test data
y_pred = model.predict(X_test_processed)
y_pred_classes = np.argmax(y_pred, axis=1)

# Map predicted numerical labels back to original labels
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
y_pred_labels = [class_names[i] for i in y_pred_classes]

# Print a few sample predictions for debugging
print("Sample predictions:")
for i in range(10):
    print(f"True label: {class_names[y_test_filtered[i]]}, Predicted: {y_pred_labels[i]}")

# Classification report
report = classification_report([class_names[i] for i in y_test_filtered], y_pred_labels, target_names=class_names)
print(report)

# Confusion matrix
cm = confusion_matrix([class_names[i] for i in y_test_filtered], y_pred_labels, labels=class_names)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()