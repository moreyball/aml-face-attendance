# Import Libraries
import tensorflow as tf
import tkinter as tk
import pickle, os, cv2
from tkinter import messagebox
from pathlib import Path

# Global Variables
FACE_CASCADE = cv2.CascadeClassifier('/Anti-Spoofing/haarcascade_frontalface_default.xml')
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 16

# Preprocessing Image
def preprocess_image(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SHAPE)
    return image

# Preprocessing Triplet Data
def preprocess_triplets(anchor, positive, negative):
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

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
model = SiameseModel(siamese_network)
# Load the model using the checkpoint
load(model, '/Anti-Spoofing/checkpoints/1121checkpoint')

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

# Face Recognize function
def recognize(img, emb_path):
    # Get the embedding of the face to be recognized
    face_embedding = embedding(tf.keras.applications.resnet50.preprocess_input(img[None, ...]))

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
            match = tf.keras.metrics.CosineSimilarity()(embeddings, face_embedding)
            print(emb_dir[index - 1][:-7], "'s similarity marks: ", match)

            # If the match score is greater than the best match score
            if best_match < match:
                # Update the best match score
                best_match = match
                # Update the best index
                best_index = index
            else:
                # Else, increment the index
                index += 1

        # If the best match score is greater than or equal to 0.97
        if best_match >= 0.97:
            # return the name of the person
            return emb_dir[best_index - 1][:-7]
        # Else if the best match score is less than 0.97
        else:
            return 'UNKNOWN_PERSON'
