import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

data_dir = 'flowers_npy'
img_size = (128, 128)
batch_size = 16
channels = 3
img_shape = (img_size[0], img_size[1], channels)


def generate_data_paths(data_dir):
    filepaths = []
    labels = []
    opened_folders = []  # List to store the names of opened folders

    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")

    folds = os.listdir(data_dir)
    if not folds:
        raise ValueError(f"No folders found in data directory {data_dir}.")

    min_count = float('inf')
    class_files = {}

    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        if not os.path.isdir(foldpath):
            continue
        opened_folders.append(fold)  # Log the folder being opened
        filelist = [os.path.join(foldpath, file) for file in os.listdir(foldpath) if file.endswith('.npy')]
        if not filelist:
            print(f"No .npy files found in {foldpath}")
        class_files[fold] = filelist
        min_count = min(min_count, len(filelist))

    for fold in class_files:
        class_files[fold] = class_files[fold][:min_count]  # Ensure uniform class distribution
        for file in class_files[fold]:
            filepaths.append(file)
            labels.append(fold)

    if not filepaths:
        raise ValueError("No valid file paths found.")

    # Log the opened folders to a file
    with open('opened_folders.log', 'w') as log_file:
        for folder in opened_folders:
            log_file.write(f"{folder}\n")

    return filepaths, labels


filepaths, labels = generate_data_paths(data_dir)
print(f"Total files found: {len(filepaths)}")  # Debugging print statement
df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Map class names to numerical labels
class_names = sorted(df['labels'].unique())
class_mapping = {label: idx for idx, label in enumerate(class_names)}
df['labels'] = df['labels'].map(class_mapping)

train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123)

# Log dataset distributions
with open('dataset_distribution.log', 'w') as log_file:
    log_file.write("Training set distribution:\n")
    log_file.write(str(train_df['labels'].value_counts()) + "\n\n")
    log_file.write("Validation set distribution:\n")
    log_file.write(str(valid_df['labels'].value_counts()) + "\n\n")
    log_file.write("Test set distribution:\n")
    log_file.write(str(test_df['labels'].value_counts()) + "\n\n")


# Function to load and preprocess images
def load_npy(file_path, label):
    image = np.load(file_path)
    image = np.resize(image, img_shape)
    image = image.astype('float32') / 255.0
    return image, label


# Data generator with modified augmentation
def data_generator(df, batch_size=32, augment=False):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        # brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    ) if augment else ImageDataGenerator()

    while True:
        df = df.sample(frac=1).reset_index(drop=True)
        for i in range(0, df.shape[0], batch_size):
            batch_df = df.iloc[i:i + batch_size]
            images, labels = zip(*[load_npy(fp, lb) for fp, lb in zip(batch_df['filepaths'], batch_df['labels'])])
            images = np.array(images, dtype='float32')
            labels = np.array(labels, dtype='int32')
            images, labels = next(datagen.flow(images, labels, batch_size=batch_size))

            # Log the labels being processed
            with open('labels_processed.log', 'a') as log_file:
                log_file.write(
                    f"Batch {i // batch_size}: {list(batch_df['labels'].value_counts().to_dict().items())}\n")

            yield images, labels


train_gen = data_generator(train_df, batch_size, augment=True)
valid_gen = data_generator(valid_df, batch_size)

# Building the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=img_shape),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')  # Use number of classes as output units
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger('flowers-training.log', separator=',', append=False)
epochs = 20

checkpoint_filepath = 'best_model.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(x=train_gen,
                    steps_per_epoch=len(train_df) // batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=valid_gen,
                    validation_steps=len(valid_df) // batch_size,
                    callbacks=[csv_logger, model_checkpoint_callback, early_stopping_callback])

model.save('flowers-model.keras')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.show()
