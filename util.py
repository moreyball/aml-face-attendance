# Import Libraries
import tensorflow as tf
import tkinter as tk
import pickle, os, cv2
import numpy as np
from tkinter import messagebox
from pathlib import Path

# Global Variables
FACE_CASCADE = cv2.CascadeClassifier('Anti-Spoofing/haarcascade_frontalface_default.xml')
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 16

# Load the base model (ResNet50) for embedding model
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", 
    input_shape=IMAGE_SHAPE + (3,), 
    include_top=False
)

# Define the embedding model
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
output = tf.keras.layers.Dense(256)(x)

# Create embedding model
embedding = tf.keras.Model(base_model.input, output, name="Embedding")

# Set all base model layers to be non-trainable
trainable = False
for layer in base_model.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

# Create the layer to compute the distance between anchor and positive embedding VS anchor and negative embedding
class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

# Create inputs for anchor, positive and negative 
anchor_input = tf.keras.layers.Input(name="anchor", shape=IMAGE_SHAPE + (3,))
positive_input = tf.keras.layers.Input(name="positive", shape=IMAGE_SHAPE + (3,))
negative_input = tf.keras.layers.Input(name="negative", shape=IMAGE_SHAPE + (3,))

# Outputs containing the distances from the anchor to the positive and negative images
distances = DistanceLayer()(
    embedding(tf.keras.applications.resnet50.preprocess_input(anchor_input)),
    embedding(tf.keras.applications.resnet50.preprocess_input(positive_input)),
    embedding(tf.keras.applications.resnet50.preprocess_input(negative_input)),
)

# Define the siamese network
siamese_network = tf.keras.Model(
    inputs=[anchor_input, positive_input, negative_input], 
    outputs=distances
)

# Create the Siamese Network model
# Computes the triplet loss using the three embeddings produced by the Siamese Network
class SiameseModel(tf.keras.Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
    
# Load the model
def load(SiameseModel, checkpoint_name):
    SiameseModel.load_weights(Path(checkpoint_name))

# Define the model
model1 = SiameseModel(siamese_network)
# Load the model using the checkpoint
load(model1, 'Anti-Spoofing/checkpoints/1130checkpoint')

model2 = tf.keras.models.load_model('Models/face_embedding_model.h5')

# Preprocessing function
def preprocess_image(img, model_selection):
    try:
        if model_selection == 1:
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, IMAGE_SHAPE)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            img = tf.expand_dims(img, axis=0)
            return img
        elif model_selection == 2:
            img = tf.image.resize(img, IMAGE_SHAPE)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = np.copy(img) / 255.0
            return img
    except Exception as e:
        error_message = f"An error occurred while processing the image: {e}\nPlease try again and make sure the bounding box is within the frame."
        messagebox.showerror("Capture Error", error_message)
        return None

# Button function
def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 20)
        )
    return button

# Image Label function
def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label

# Text Label function
def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label

# Entry Text function
def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt

# Message Box function
def msg_box(title, description):
    # Create a messagebox
    messagebox.showinfo(title, description)

# Face Detection function
def detect(frame):
    # Convert the captured frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the Haar Cascade classifier to detect faces
    faces = FACE_CASCADE.detectMultiScale(gray_frame, 1.3, 5)

    return faces

# Draw Bounding Box function
def bounding_box(frame):
    # Detect Face
    faces = detect(frame)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        margin = 50
        x -= margin
        y -= margin * 2
        w += margin * 2
        h += margin * 3
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Face Recognize function using Cosine Similarity
def recognize(img, emb_path, model_selection):
    # Initialize the model selection variable
    # 1 = Metric Learning
    # 2 = Classification
    face_embedding = threshold = 0

    if model_selection == 1:            
        face_embedding = embedding(img) # Extract the embeddings
        threshold = 0.97
    elif model_selection == 2:
        face_embedding = model2.predict(img) # Extract the embeddings
        threshold = 0.9985

    # If no face is detected, then return 'NO_PERSON_FOUND'
    if len(face_embedding) == 0:
        return 'NO_PERSON_FOUND'
    # Else if face is detected, then proceed to recognize the face
    else:
        # Load the database
        emb_dir = sorted(os.listdir(emb_path))

        # Initialize the index and match score variables
        index = match = best_match = best_index = 0

        # While the index is less than the length of the database
        while index < len(emb_dir):
            # Get the selected embedding path of the database
            path_ = os.path.join(emb_path, emb_dir[index])

            # Load the embeddings from the database
            file = open(path_, 'rb')
            embeddings = pickle.load(file)

            # Compute the cosine similarity between the face to be recognized and the embeddings from the database
            match = tf.keras.metrics.CosineSimilarity()(embeddings, face_embedding).numpy()
            print(emb_dir[index][:-7], "'s similarity marks: ", match)

            # If the match score is greater than the best match score
            if best_match < match:
                # Update the best match score
                best_match = match
                # Update the best index
                best_index = index

            # Increment the index
            index += 1

        # If the best match score is greater than or equal to 0.5
        if best_match >= threshold:
            # return the name of the person
            return emb_dir[best_index][:-7]
        # Else if the best match score is less than 0.5
        else:
            return 'UNKNOWN_PERSON'
# Face Recognize function using euclidean distance   
def recognize2(img, emb_path, model_selection):
    # Initialize the model selection variable
    # 1 = Metric Learning
    # 2 = Classification
    face_embedding = threshold = 0

    if model_selection == 1:            
        face_embedding = embedding(img) # Extract the embeddings
        threshold = 1.5  # Example threshold for Euclidean distance
    elif model_selection == 2:
        face_embedding = model2.predict(img) # Extract the embeddings
        threshold = 12  # Example threshold for Euclidean distance

    # If no face is detected, then return 'NO_PERSON_FOUND'
    if len(face_embedding) == 0:
        return 'NO_PERSON_FOUND'
    # Else if face is detected, then proceed to recognize the face
    else:
        # Load the database
        emb_dir = sorted(os.listdir(emb_path))

        best_match_distance = float('inf')  # Set initial distance to infinity
        best_match_name = None
        # While the index is less than the length of the database
        for index, emb in enumerate(emb_dir):
            path_ = os.path.join(emb_path, emb)
            file = open(path_, 'rb')
            embeddings = pickle.load(file)

            # Calculate Euclidean distance
            distance = tf.norm(embeddings - face_embedding, ord='euclidean')

            print(emb[:-7], "'s distance: ", distance)


            # Update the best match based on distance
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_name = emb[:-7]  
        print('Best match name: ', best_match_name)
        # If the best distance is less than the threshold
        if best_match_distance < threshold:
            # return the name of the person
            return best_match_name
        # Else if the best distance is greater than or equal to the threshold
        else:
            return 'UNKNOWN_PERSON'
