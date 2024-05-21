import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import json

# Constants
DATA_DIR = 'data'
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
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=6)

# Predict on test data
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

test_generator = test_datagen.flow_from_directory(
    directory=DATA_DIR,
    target_size=(128, 128),
    batch_size=4,
    class_mode=None,
    shuffle=False
)

# Make predictions
predictions = model.predict(test_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Save predictions
predictions_dict = {os.path.basename(filepath): int(pred) for filepath, pred in zip(test_generator.filenames, predicted_classes)}
with open(PREDICTIONS_PATH, 'w') as f:
    json.dump({'target': predictions_dict}, f)

print(f"Predictions saved to {PREDICTIONS_PATH}")