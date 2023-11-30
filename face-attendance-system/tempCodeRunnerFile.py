# Import Libraries
import os, pickle, datetime, cv2
import tkinter as tk
import tensorflow as tf
import util
from PIL import Image, ImageTk
from test import test

# Face Attendance System
class Face_Attendance_System:
    # Initialize the Face Attendance System
    def __init__(self):
        # Create the main window
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100") # width x height + x_offset + y_offset

        # Create Login Button
        self.login_button_main_window = util.get_button(self.main_window, 'LOGIN', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        # Create Register Button
        self.register_button_main_window = util.get_button(self.main_window, 'REGISTER', 'gray', self.register, fg='black')
        self.register_button_main_window.place(x=750, y=300)

        # Create Webcam Label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        # Add Webcam to Label
        self.add_webcam(self.webcam_label)

        # Create Embedding Space
        self.emb_dir = 'Embedding_Space'
        if not os.path.exists(self.emb_dir):
            os.mkdir(self.emb_dir)

        # Create Log File
        self.log_path = 'Attendance_List.txt'

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
    def login(self):
        # Anti-Spoofing System
        label = test(
                image=self.most_recent_capture_arr,
                model_dir='/Anti-Spoofing/resources/anti_spoof_models',
                device_id=0
                )

        # Check whether the user is a spoofer
        # If the user is not a spoofer, then perform face recognition
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

                    # Resize the cropped face to a standard size (e.g., 200x200)
                    cropped_face = cv2.resize(cropped_face, (200, 200))

                    # Match the extracted embeddings with the saved face embeddings in the database
                    name = util.recognize(cropped_face, self.emb_dir)

                    # Check the recognition result
                    # If the recognized user is not in the database, then show an error message
                    if name in ['UNKNOWN_PERSON']:
                        util.msg_box('!!UNKNOWN PERSON!!', 'UNKNOWN PERSON DETECTED\nPLEASE PROCEED TO REGISTER')
                    elif name in ['NO_PERSON_FOUND']:
                        util.msg_box('!!NO PERSON FOUND!!', 'NO PERSON DETECTED\nPLEASE TRY AGAIN')
                    else:
                        util.msg_box('ATTENDANCE MADE', 'HELLO, {}.'.format(name))
                        with open(self.log_path, 'a') as f:
                            f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                            f.close()
        # If the user is a spoofer, then show an error message
        else:
            util.msg_box('ANTI-SPOOFING ACTIVATED', 'SPOOFER DETECTED\nPLEASE TRY AGAIN')

    # Register Function
    def register(self):
        # Create Register Window
        self.register_window = tk.Toplevel(self.main_window)
        self.register_window.geometry("1200x520+370+120") # width x height + x_offset + y_offset

        # Create Accept Button
        self.accept_button_register_window = util.get_button(self.register_window, 'CONFIRM', 'green', self.accept_register)
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
    def accept_register(self):
        # Anti-Spoofing System
        label = test(
                image=self.register_capture,
                model_dir='/Anti-Spoofing/resources/anti_spoof_models',
                device_id=0
                )
        
        # Check whether the user is a spoofer
        # If the user is not a spoofer, then perform face recognition
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

                # Resize the cropped image to a standard size (e.g., 200x200)
                cropped_img = cv2.resize(cropped_img, (200, 200))

                # Extract the embeddings
                embeddings = util.embedding(tf.keras.applications.resnet50.preprocess_input(cropped_img[None, ...]))

                # Save the embeddings to the database
                file = open(os.path.join(self.emb_dir, '{}.pickle'.format(name)), 'wb')
                # Dump the embeddings to the file
                pickle.dump(embeddings, file)

                # Show a success message
                util.msg_box('REGISTERED!!', '{} REGISTER SUCCESS'.format(name))

                # Destroy Register Window
                self.register_window.destroy()
                return
        # If the user is a spoofer, then show an error message
        else:
            util.msg_box('ANTI-SPOOFING ACTIVATED', 'SPOOFER DETECTED\nPLEASE TRY AGAIN')

# Main Function
if __name__ == "__main__":
    Face_Attendance_System = Face_Attendance_System()
    Face_Attendance_System.start()