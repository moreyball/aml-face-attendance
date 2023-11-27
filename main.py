import os.path
import datetime
import pickle

import tkinter as tk
import cv2
from PIL import Image, ImageTk

import util
from test import test

import tensorflow as tf

face_cascade = cv2.CascadeClassifier('C:/Users/liang/Downloads/COS30082/Silent-Face-Anti-Spoofing/haarcascade_frontalface_default.xml')

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir='C:/Users/liang/Downloads/COS30082/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
                device_id=0
                )

        if label == 1:
                # Capture the most recent frame from the webcam
                ret, frame = self.cap.read()

                # Convert the captured frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Use the Haar Cascade classifier to detect faces
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

                # Check if a face is detected
                if len(faces) > 0:
                    # Take the first detected face (you can modify this logic based on your requirements)
                    x, y, w, h = faces[0]

                    # Crop the image based on the bounding box of the detected face
                    cropped_face = frame[y:y+h, x:x+w]

                    # Resize the cropped face to a standard size (e.g., 200x200)
                    cropped_face = cv2.resize(cropped_face, (200, 200))

                    # Match the extracted embeddings with the saved face embeddings in the database
                    name = util.recognize(cropped_face, self.db_dir)

                    # Check the recognition result
                    if name in ['unknown_person']:
                        util.msg_box('Ups...', 'Unknown user. Please register a new user or try again.')
                    elif name in ['no_persons_found']:
                        util.msg_box('Ups...', 'No person found. Please try again.')
                    else:
                        util.msg_box('Welcome back!', 'Welcome, {}.'.format(name))
                        with open(self.log_path, 'a') as f:
                            f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                            f.close()

                else:
                    # If no face is detected, show an error message
                    util.msg_box('Error', 'No face detected. Please try again.')

            # name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            # if name in ['unknown_person']:
            #     util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            # elif name in ['no_persons_found']:
            #     util.msg_box('Ups...', 'No person found. Please try again.')
            # else:
            #     util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
            #     with open(self.log_path, 'a') as f:
            #         f.write('{},{},in\n'.format(name, datetime.datetime.now()))
            #         f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    def logout(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir='C:/Users/liang/Downloads/COS30082/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
                device_id=0
                )

        if label == 1:

            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Hasta la vista !', 'Goodbye, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')


    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()
        
        # Convert the captured frame to grayscale
        gray_frame = cv2.cvtColor(self.register_new_user_capture, cv2.COLOR_BGR2GRAY)
        
        # Use the Haar Cascade classifier to detect faces
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        if len(faces) == 0:
            util.msg_box('Error', 'No face detected. Please try again.')
            return

        # Take the first detected face (you can modify this logic based on your requirements)
        x, y, w, h = faces[0]

        # Crop the image based on the bounding box of the detected face
        cropped_img = self.register_new_user_capture[y:y+h, x:x+w]

        # Resize the cropped image to a standard size (e.g., 200x200)
        cropped_img = cv2.resize(cropped_img, (200, 200))

        # Use the cropped image to extract face embeddings
        img = tf.image.resize(cropped_img, (200, 200))
        model = tf.keras.models.load_model("Silent-Face-Anti-Spoofing/single_image_embedding_model.h5")
        embeddings = model.predict(tf.keras.applications.resnet50.preprocess_input(img[None, ...]))

        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()