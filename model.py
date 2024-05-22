import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Constants
DATA_DIR = 'data'
TEST_DIR = 'data/test'
LABELS_PATH = 'data/labels_path.csv'
PREDICTIONS_PATH = 'predictions/predictions.json'

# Load data
data_df = pd.read_csv(LABELS_PATH)
data_df['label'] = data_df['label'].astype(str)

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=False,
    rotation_range=360,
    zoom_range=0.12
)
train_generator = train_datagen.flow_from_dataframe(
    data_df,
    directory=DATA_DIR,
    x_col='path',
    y_col='label',
    target_size=(128, 128),
    batch_size=4,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_dataframe(
    data_df,
    directory=DATA_DIR,
    x_col='path',
    y_col='label',
    target_size=(128, 128),
    batch_size=4,
    class_mode='binary',
    subset='validation'
)

# Load pre-trained model and fine-tune
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(512, activation='relu')(x)
#x = BatchNormalization()(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=6)

# Preprocess
image_paths = [os.path.join(TEST_DIR, img_name) for img_name in os.listdir(TEST_DIR) if img_name.endswith(('.jpg', '.jpeg'))]
def preprocess_image(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        out.append(x)
    return np.array(out)
preprocessed_images = preprocess_image(image_paths)

# Make predictions
predictions = model.predict(preprocessed_images)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Save predictions
predictions_dict = {os.path.basename(filepath): int(pred) for filepath, pred in zip(image_paths, predicted_classes)}
with open(PREDICTIONS_PATH, 'w') as f:
    json.dump({'target': predictions_dict}, f)

print(f"Predictions saved to {PREDICTIONS_PATH}")