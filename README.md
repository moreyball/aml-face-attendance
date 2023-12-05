# aml-face-attendance
To ensure that the system works, please install all the dependencies in the requirements.txt file. To do so, run the following command in the terminal.

```bash
pip install -r requirements.txt
```

## Anti-spoofing
For windows, in order for anti-spoofing to work, set PYTHONPATH to the directory of anti-spoofing folder.
1. Go to edit system environment variables in settings
2. Add new system variable .
3. Variable name is PYTHONPATH, value is directory of anti-spoofing folder

## Models

Due to github upload file restrictions, the HDF5 models is not in the repo. Please download the files from the following [link]((https://drive.google.com/drive/folders/1oxgZgjsnpTs-2LTgrqR2T5DX8j9_AyxI?usp=drive_link)) and place it in the Models folder.

## How to run the system
1. Run the main.py file
2. Choose which model to run (classification or metric learning)
3. Register to add your face to the database
4. Login to take your attendance.






