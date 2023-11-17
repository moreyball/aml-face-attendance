import os.path
import datetime
import pickle

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition

import util
from test import test 


class App:
    def __init__(self):
       # Initialize the main window
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")
        # Buttons for different actions
        self.login_button_main_window = util.get_button(self.main_window, 'Take Attendance', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)
        
        self.logout_button_main_window = util.get_button(self.main_window, 'Logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'Register New User', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)
        # Label to display webcam feed
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)
        # Create a directory for the database if it doesn't exist
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        # Path to the log file
        self.log_path = './log.txt'
    
    
    def add_webcam(self, label):  # label is a tkinter label
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0) # 0 = default camera

        self._label = label
        self.process_webcam()

    def process_webcam(self): #put the webcam feed into the tkinter label
        ret, frame = self.cap.read()

        self.most_recent_capture_arr = frame # save the most recent capture as an numpy array
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB) # convert to RGB
        self.most_recent_capture_pil = Image.fromarray(img_) # convert to PIL format
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil) # convert to ImageTk format
        self._label.imgtk = imgtk # anchor imgtk so it does not be deleted by garbage-collector
        self._label.configure(image=imgtk) # show the image

        self._label.after(20, self.process_webcam) # call the same function after 20 milliseconds

    def login(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir=r"C:\Users\Winston\Documents\GitHub\Automated-Attendance-System-with-Face-Recognition\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
                device_id=0
                )

        if label == 1:

            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    def logout(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir=r"C:\Users\Winston\Documents\GitHub\Automated-Attendance-System-with-Face-Recognition\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
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

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Retake Picture', 'brown', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please\ninput a username:')
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
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]
        print("Embeddings shape:", embeddings.shape)
        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
