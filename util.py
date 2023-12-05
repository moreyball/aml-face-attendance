# Import Libraries
import tensorflow as tf
import tkinter as tk
import pickle, os, cv2
import numpy as np
from tkinter import messagebox
from pathlib import Path

# Global Variables
FACE_CASCADE = cv2.CascadeClassifier('Anti_Spoofing/haarcascade_frontalface_default.xml')
IMAGE_SHAPE = (224, 224)

# Load the model (Metric Learning)
model1 = tf.keras.models.load_model('Models/metric_learning_embedding_model.h5')

# Load the model (Classification)
model2 = tf.keras.models.load_model('Models/classification_embedding_model.h5')

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
        font=('Gill Sans MT Bold', 20)
        )
    return button

# Function to create radio buttons
def get_radio_button(window, text, variable, value):
    radio_button = tk.Radiobutton(
        window,
        text=text,
        variable=variable,
        value=value,
        font=('Gill Sans MT Bold', 15), 
        justify="left"
    )
    return radio_button

# Image Label function
def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label

# Text Label function
def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("Gill Sans MT Bold", 21), justify="left")
    return label

# Entry Text function
def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Gill Sans MT Bold", 32))
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
def recognize1(img, emb_path, model_selection):
    # Initialize the model selection variable
    # 1 = Metric Learning
    # 2 = Classification
    face_embedding = threshold = 0

    if model_selection == 1:            
        face_embedding = model1.predict(img) # Extract the embeddings
        threshold = 0.9994136095046997 # Best threshold for Cosine Similarity
    elif model_selection == 2:
        face_embedding = model2.predict(img) # Extract the embeddings
        threshold = 0.42888376116752625 # Best threshold for Cosine Similarity

    # If no face is detected, then return 'NO_PERSON_FOUND'
    if len(face_embedding) == 0:
        return 'NO_PERSON_FOUND', 0
    # Else if face is detected, then proceed to recognize the face
    else:
        # Load the database
        emb_dir = sorted(os.listdir(emb_path))

        # Initialize the index and similarity_score variables
        similarity_score = best_similarity_score = 0
        person = None

        # While the index is less than the length of the database
        for _, emb in enumerate(emb_dir):
            # Get the selected embedding path of the database
            path_ = os.path.join(emb_path, emb)
            file = open(path_, 'rb')
            embeddings = pickle.load(file)

            # Compute the cosine similarity between the face to be recognized and the embeddings from the database
            similarity_score = tf.keras.metrics.CosineSimilarity()(embeddings, face_embedding).numpy()
            print(emb[:-7], "'s similarity marks: ", similarity_score)

            # If the similarity_score is greater than the best similarity_score
            if best_similarity_score < similarity_score:
                # Update the best similarity_score and person
                best_similarity_score = similarity_score
                person = emb[:-7]

        # If the best similarity_score is greater than or equal to the threshold
        if best_similarity_score >= threshold:
            return person, best_similarity_score
        else:
            return 'UNKNOWN_PERSON', best_similarity_score

# Face Recognize function using euclidean distance   
def recognize2(img, emb_path, model_selection):
    # Initialize the model selection variable
    # 1 = Metric Learning
    # 2 = Classification
    face_embedding = threshold = 0

    if model_selection == 1:            
        face_embedding = model1.predict(img) # Extract the embeddings
        threshold = 1 / 0.35788866877555847 - 1  #Threshold for Euclidean distance / 1.79416502
    elif model_selection == 2:
        face_embedding = model2.predict(img) # Extract the embeddings
        threshold = 15   # (1 / 0.033828988671302795 - 1)  # Threshold for Euclidean distance / 28.5604462

    # If no face is detected, then return 'NO_PERSON_FOUND'
    if len(face_embedding) == 0:
        return 'NO_PERSON_FOUND', 0
    # Else if face is detected, then proceed to recognize the face
    else:
        # Load the database
        emb_dir = sorted(os.listdir(emb_path))
        
        # Initialize the index and similarity_score variables
        best_euclidean_distance = float('inf')  # Set initial distance to infinity
        person = None

        for _, emb in enumerate(emb_dir):
            path_ = os.path.join(emb_path, emb)
            file = open(path_, 'rb')
            embeddings = pickle.load(file)

            # Calculate Euclidean distance
            euclidean_distance = tf.norm(embeddings - face_embedding, ord='euclidean').numpy()
            print(emb[:-7], "'s euclidean_distance: ", euclidean_distance)

            # If the euclidean_distance is less than the best euclidean_distance
            if euclidean_distance < best_euclidean_distance:
                # Update the best similarity_score and person
                best_euclidean_distance = euclidean_distance
                person = emb[:-7]  
        print('---------------------------------------')
        # If the best euclidean_distance is less than the threshold
        if best_euclidean_distance < threshold:
            return person, best_euclidean_distance
        else:
            return 'UNKNOWN_PERSON', best_euclidean_distance
