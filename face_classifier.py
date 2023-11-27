#%%
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
print(tf.__version__)

#%%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#%%
train_data_dir = r'C:\Users\102774145\Desktop\classification_data\classification_data\train_data'
test_data_dir = r'C:\Users\102774145\Desktop\classification_data\classification_data\test_data'
val_data_dir = r'C:\Users\102774145\Desktop\classification_data\classification_data\val_data'

#%%
image_size = (224, 224)
epochs = 15
batch_size = 32
num_class = len(os.listdir(train_data_dir))

#%%
train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255.0,
    horizontal_flip = True,
    rotation_range = 10
    )
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
    )
val_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = False
    )
test_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = False
    )

#%%
base_model = ResNet50(weights = 'imagenet', input_shape=image_size + (3,), include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(256, activation = 'relu')(x)

x = Dropout(0.5)(x)
predictions = Dense(num_class, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#%%
model.summary()
#%%
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#%%
trained_model = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = val_generator,
    validation_steps = val_generator.samples // batch_size,
    epochs = epochs
    )

#%%
model.save(r'C:\Users\102774145\Desktop\face_classification_model.h5')

#%%
#Show Accuracy
train_accuracy = trained_model.history['accuracy'][-1] * 100
val_accuracy = trained_model.history['val_accuracy'][-1] * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {val_accuracy:.2f}%")

#%%
import matplotlib.pyplot as plt
#Show Loss Overtime Graph
plt.plot(trained_model.history['loss'], label = 'Training Loss')
plt.plot(trained_model.history['val_loss'], label = 'Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
#Show Accuracy Overtime Graph
plt.plot(trained_model.history['accuracy'], label = 'Training Accuracy')
plt.plot(trained_model.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Accuracy  Overtime Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%%
from tensorflow.keras.models import load_model
model_path = r"C:\Users\102774145\Desktop\face_classification_model.h5"
trained_model = load_model(model_path)

#%%
from sklearn.metrics import confusion_matrix, classification_report
testing_predictions = model.predict(test_generator)
predicted_labels = np.argmax(testing_predictions, axis=1)

true_labels = test_generator.classes

confusion_mat = confusion_matrix(true_labels, predicted_labels)

class_names = list(test_generator.class_indices.keys())
classification_rep = classification_report(true_labels, predicted_labels, target_names=class_names, zero_division = 1)

print("Confusion Matrix:")
print(confusion_mat)

print("\nClassification Report:")
print(classification_rep)

#%%
import matplotlib.pyplot as plt
import random

def display_images_with_predictions(test_generator, model, num_images=25):
    # Get class names
    class_names = list(test_generator.class_indices.keys())

    # Get a batch of random test images
    test_batch, test_labels = next(test_generator)

    # Make predictions on the test batch
    predictions = model.predict(test_batch)

    plt.figure(figsize=(12, 12))

    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_batch[i])
        plt.axis('off')

        predicted_class = class_names[np.argmax(predictions[i])]
        actual_class = class_names[np.argmax(test_labels[i])]

        plt.title(f'Predicted: {predicted_class}\nActual: {actual_class}')

    plt.show()

# Display 25 random images with predictions
display_images_with_predictions(test_generator, model, num_images=25)