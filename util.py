import os
import pickle

import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
import face_recognition

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


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img, db_path):
    # it is assumed there will be at most 1 match in the db

    img = tf.image.resize(img, (200,200))
    model = tf.keras.models.load_model("Silent-Face-Anti-Spoofing/single_image_embedding_model.h5")
    embeddings_unknown = model.predict(tf.keras.applications.resnet50.preprocess_input(img[None,...]))
    
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    db_dir = sorted(os.listdir(db_path))

    j = 0
    match = 0
    while not match and j < len(db_dir):
        path_ = os.path.join(db_path, db_dir[j])

        file = open(path_, 'rb')
        embeddings = pickle.load(file)

        match = tf.keras.losses.cosine_similarity(embeddings, embeddings_unknown).numpy()
        print(match)
        j += 1
        if match >= 0.2:
            break

    if match >= 0.2:
        return db_dir[j - 1][:-7]
    else:
        return 'unknown_person'
