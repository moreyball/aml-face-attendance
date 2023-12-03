# Import Libraries
import os, pickle, datetime, cv2
import tkinter as tk
import tensorflow as tf
import numpy as np
import util
from PIL import Image, ImageTk
from test import test

def model_selection():
    model = 2 # Default Value

    # Model 1 Selection
    def model_selection_1():
        nonlocal model
        main_window.destroy()  # Close the main window
        model = 1

    # Model 2 Selection
    def model_selection_2():
        nonlocal model
        main_window.destroy()  # Close the main window
        model = 2

    # Create the main window
    main_window = tk.Tk()
    main_window.geometry("550x400+500+200")

    # Create Model 1 Button
    model1_button = util.get_button(main_window, 'MODEL 1\n(METRIC LEARNING)', 'red', model_selection_1)
    model1_button.place(x=100, y=100)

    # Create Model 2 Button
    model2_button = util.get_button(main_window, 'MODEL 2\n(CLASSIFICATION)', 'blue', model_selection_2)
    model2_button.place(x=100, y=200)

    # Start the main window
    main_window.mainloop()

    return model


# Face Attendance System
class Face_Attendance_System:
    # Initialize the Face Attendance System
    def __init__(self, model_selection):
        # Create the main window
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100") # width x height + x_offset + y_offset

        if model_selection == 1:
            # A Display Label
            self.display_label = util.get_text_label(self.main_window, 'FACE ATTENDANCE SYSTEM\n (METRIC LEARNING)')
            self.display_label.place(x=750, y=50)
        elif model_selection == 2:
            # A Display Label
            self.display_label = util.get_text_label(self.main_window, 'FACE ATTENDANCE SYSTEM\n (CLASSIFICATION)')
            self.display_label.place(x=750, y=50)

        # Create Login Button
        self.login_button_main_window = util.get_button(self.main_window, 'LOGIN', 'green', lambda: self.login(model_selection))
        self.login_button_main_window.place(x=750, y=200)

        # Create Register Button
        self.register_button_main_window = util.get_button(self.main_window, 'REGISTER', 'gray', lambda: self.register(model_selection), fg='black')
        self.register_button_main_window.place(x=750, y=300)

        # Create Webcam Label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        # Add Webcam to Label
        self.add_webcam(self.webcam_label)

        # Create Embedding Space
        self.emb1_dir = 'Embedding_Space_Model1'
        if not os.path.exists(self.emb1_dir):
            os.mkdir(self.emb1_dir)
        self.emb2_dir = 'Embedding_Space_Model2'
        if not os.path.exists(self.emb2_dir):
            os.mkdir(self.emb2_dir)

        # Create Log File
        self.log1_path = 'Attendance_List_Model1.txt'
        self.log2_path = 'Attendance_List_Model2.txt'

    # Add Webcam to Label
    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            # Capture the most recent frame from the webcam
            self.cap = cv2.VideoCapture(0)

        # Set Self Label
        self._label = label
        # Process Webcam
        self.process_webcam()

    # Process Webcam
    def process_webcam(self):
        # Capture the most recent frame from the webcam
        ret, frame = self.cap.read()

        # Detect the faces
        util.bounding_box(frame)

        # Most recent capture array = frame
        self.most_recent_capture_arr = frame
        # Convert the most recent capture array to Color
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        # Most recent capture PIL = Image from array
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        # Set Self Label
        self._label.imgtk = imgtk
        # Configure the label
        self._label.configure(image=imgtk)

        # Call the process webcam function again after 20 milliseconds
        self._label.after(20, self.process_webcam)

    # Login Function
    def login(self, model_selection):
        # Anti-Spoofing System
        label = 1
        
        if label == 1:
                # Capture the most recent frame from the webcam
                ret, frame = self.cap.read()
                
                # Detect the faces
                faces = util.detect(frame)

                # Check if a face is detected
                # If no face is detected, then show an error message
                if len(faces) == 0:
                    util.msg_box('!!ERROR!!', 'NO FACE DETECTED\nPLEASE TRY AGAIN')
                    return
                # Else if face is detected, then proceed to recognize the user
                else:
                    # Take the first detected face (you can modify this logic based on your requirements)
                    x, y, w, h = faces[0]

                    margin = 50
                    x -= margin
                    y -= margin * 2
                    w += margin * 2
                    h += margin * 3

                    # Crop the image based on the modified bounding box
                    cropped_face = frame[y:y + h, x:x + w]

                    # Preprocess the cropped face
                    cropped_face = util.preprocess_image(cropped_face, model_selection)
                    print(cropped_face.shape)
                    if cropped_face is None:
                        return  # Exit the function as the image preprocessing failed

                    # Match the extracted embeddings with the saved face embeddings in the database
                    if model_selection == 1:
                        name = util.recognize2(cropped_face, self.emb1_dir, model_selection)
                    elif model_selection == 2:
                        name = util.recognize2(cropped_face, self.emb2_dir, model_selection)

                    # Check the recognition result
                    # If the recognized user is not in the database, then show an error message
                    if name in ['UNKNOWN_PERSON']:
                        util.msg_box('!!UNKNOWN PERSON!!', 'UNKNOWN PERSON DETECTED\nPLEASE PROCEED TO REGISTER')
                    # Else if no face is detected, then show an error message
                    elif name in ['NO_PERSON_FOUND']:
                        util.msg_box('!!NO PERSON FOUND!!', 'NO PERSON DETECTED\nPLEASE TRY AGAIN')
                    # Else the recognized user is in the database, then show a success message
                    else:
                        util.msg_box('ATTENDENCE MADE', 'HELLO, {}.'.format(name))
                        if model_selection == 1:
                            with open(self.log1_path, 'a') as f:
                                f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                                f.close()
                        elif model_selection == 2:
                            with open(self.log2_path, 'a') as f:
                                f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                                f.close()
        # If the user is a spoofer, then show an error message
        else:
            util.msg_box('ANTI-SPOOFING ACTIVATED', 'SPOOFER DETECTED\nPLEASE TRY AGAIN')
            # Destroy Register Window
            self.register_window.destroy()

    # Register Function
    def register(self, model_selection):
        # Create Register Window
        self.register_window = tk.Toplevel(self.main_window)
        self.register_window.geometry("1200x520+370+120") # width x height + x_offset + y_offset

        # Create Accept Button
        self.accept_button_register_window = util.get_button(self.register_window, 'CONFIRM', 'green', lambda: self.accept_register(model_selection))
        self.accept_button_register_window.place(x=750, y=300)

        # Create Try Again Button
        self.try_again_button_register_window = util.get_button(self.register_window, 'TRY AGAIN', 'red', self.try_again_register)
        self.try_again_button_register_window.place(x=750, y=400)

        # Create Capture Label
        self.capture_label = util.get_img_label(self.register_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        # Add Webcam to Label
        self.add_img_to_label(self.capture_label)

        # Create Entry Text
        self.entry_text_register = util.get_entry_text(self.register_window)
        self.entry_text_register.place(x=750, y=150)

        # Create Text Label
        self.text_label_register = util.get_text_label(self.register_window, 'USERNAME: ')
        self.text_label_register.place(x=750, y=100)

    # Try Again Register Function
    def try_again_register(self):
        # Destroy Register Window
        self.register_window.destroy()

    # Add Image to Label
    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        # Set Self Label
        label.imgtk = imgtk
        # Configure the label
        label.configure(image=imgtk)

        # Set Register Capture
        self.register_capture= self.most_recent_capture_arr.copy()

    # Start Function
    def start(self):
        self.main_window.mainloop()

    # Accept Register Function
    def accept_register(self, model_selection):
        # Anti-Spoofing System
        label = 1
        
        if label == 1:
            # Get Name from Entry Text
            name = self.entry_text_register.get(1.0, "end-1c").strip()
            
            faces = util.detect(self.register_capture)

            # Check if a face is detected
            # If no face is detected, then show an error message
            if len(faces) == 0:
                util.msg_box('!!ERROR!!', 'NO FACE DETECTED\nPLEASE TRY AGAIN')
                return
            # Else if face is detected, then proceed to register the user
            else:
                # Take the first detected face
                x, y, w, h = faces[0] # x, y, width, height

                # Add a margin on the face detected
                margin = 50
                x -= margin
                y -= margin * 2
                w += margin * 2
                h += margin * 3

                # Crop the image based on the modified bounding box
                cropped_img = self.register_capture[y:y + h, x:x + w]

                cropped_img = util.preprocess_image(cropped_img, model_selection)

                if cropped_img is None:
                    return  # Exit the function as the image preprocessing failed

                if model_selection == 1:            
                    embeddings = util.embedding(cropped_img) # Extract the embeddings
                    file = open(os.path.join(self.emb1_dir, '{}.pickle'.format(name)), 'wb') # Open the database
                    pickle.dump(embeddings, file) # Save the embeddings to the database
                elif model_selection == 2:
                    embeddings = util.model2.predict(cropped_img) # Extract the embeddings
                    file = open(os.path.join(self.emb2_dir, '{}.pickle'.format(name)), 'wb') # Open the database
                    pickle.dump(embeddings, file) # Save the embeddings to the database

                # Show a success message
                util.msg_box('REGISTERED!!', '{} REGISTER SUCCESS'.format(name))

                # Destroy Register Window
                self.register_window.destroy()
                return
        # If the user is a spoofer, then show an error message
        else:
            util.msg_box('ANTI-SPOOFING ACTIVATED', 'SPOOFER DETECTED\nPLEASE TRY AGAIN')
            # Destroy Register Window
            self.register_window.destroy()

# Main Function
if __name__ == "__main__":
    # Model Selection
    model = model_selection()
    Face_Attendance_System = Face_Attendance_System(model)
    Face_Attendance_System.start()