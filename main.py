#Face Dectection system by Rohan Gazi

# Imports

# Open cv2 (Open Source Computer Vision Library) library from https://opencv.org 
import cv2
# Tkinter (Standard Python interface to the Tk GUI toolkit) library from https://docs.python.org/3/library/tkinter.html 
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import tkinter.filedialog as filedialog

# PIL (Python Imaging Library) library from https://pillow.readthedocs.io/en/stable/
from PIL import Image, ImageTk
# Cryptography library from https://cryptography.io/en/latest/fernet/
from cryptography.fernet import Fernet
import base64
import os
import json
import numpy as np

# Encryption code adapted from https://blog.bytescrum.com/encrypting-and-decrypting-data-with-fernet-in-python#%20Function%20to%20load%20or%20generate%20the%20encryption%20keydef%20load_or_generate_key():key_file_path%20=%20'encryption_key.json'if%20os.path.exists(key_file_path):with%20open(key_file_path,%20'r')%20as%20key_file:key_data%20=%20json.load(key_file)key%20=%20base64.urlsafe_b64decode(key_data['key'])else:key%20=%20Fernet.generate_key()with%20open(key_file_path,%20'w')%20as%20key_file:key_data%20=%20{'key':%20base64.urlsafe_b64encode(key).decode()}json.dump(key_data,%20key_file)return%20key
# Function to load or generate the encryption key
def load_or_generate_key():
    key_file_path = 'key.json'
    if os.path.exists(key_file_path):
        with open(key_file_path, 'r') as key_file:
            key_data = json.load(key_file)
            key = base64.urlsafe_b64decode(key_data['key'])
    else:
        key = Fernet.generate_key()
        with open(key_file_path, 'w') as key_file:
            key_data = {'key': base64.urlsafe_b64encode(key).decode()}
            json.dump(key_data, key_file)
    return key

# Function to encrypt data
def encrypt_data(data, key):
    cipher = Fernet(key)
    return cipher.encrypt(data.encode())

# Function to decrypt data
def decrypt_face_data(data, key):
    cipher = Fernet(key)
    decrypted_data = cipher.decrypt(data)
    nparr = np.frombuffer(decrypted_data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr

# Function to decrypt data
def decrypt_data(data, key):
    cipher = Fernet(key)
    decrypted_data = cipher.decrypt(data)
    return decrypted_data

# Load or generate the encryption key
key = load_or_generate_key()

def start_face_detection():

    # User interface button styling code adapted from a solution on StackOverflow: https://stackoverflow.com/questions/27347981/how-to-change-the-color-of-ttk-button
    # Change colour of button to black
    style = ttk.Style()
    style.theme_use('alt')
    style.configure('Custom.TButton', background='black', foreground='white', borderwidth=1, focusthickness=3, focuscolor='none',width = 20)
    style.map('Custom.TButton', background=[('active','black')])

    # Close main menu frame
    main_menu_frame.destroy()

    # New frame for face detection
    face_detection_frame = ttk.Frame(root)
    face_detection_frame.pack()

    # Label for displaying the video feed
    label = ttk.Label(face_detection_frame)
    label.pack()

    # Open video capture
    cap = cv2.VideoCapture(0)

    # Load pre-trained face detection models from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Bool to check if privacy mode is active
    # Variable to track privacy mode state
    privacy_mode_active = False

    # Function to toggle privacy mode
    def toggle_privacy_mode():
        nonlocal privacy_mode_active
        privacy_mode_active = not privacy_mode_active

    # Exit button function
    def stop_capture():
        # Release video capture and writer
        cap.release()

        # Destroy the face detection frame
        face_detection_frame.destroy()
        
        # Recreate the main menu frame
        create_main_menu()

    # Real-time face image capture code adapted from a solution provided on Stack Overflow: https://stackoverflow.com/questions/41688849/opencv-python-how-to-save-name-of-the-recognized-face-from-face-recognition-pro
    # Capture the face of people function
    def capture_frame():
        nonlocal privacy_mode_active

        if not privacy_mode_active:
            # Capture the current frame
            ret, frame = cap.read()
            if ret:
                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Prompt user for a name or identifier for each captured face using GUI
            if len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):

                    # Prompt user for a name or identifier for each captured face using a dialog box
                    try:
                        face_name = simpledialog.askstring(f"Enter a name for face {i + 1}", "Name:")
                        if face_name is not None:
                            # Get the information displayed in the features label
                            features_text = features_label['text']

                            # Save the captured face image and face data
                            save_face_image(face_name, frame[y:y + h, x:x + w], features_text)
                    except KeyboardInterrupt:
                        stop_capture()
            else:
                print("No faces detected in this frame.")

    # Create a box to display real time detected features
    features_label = ttk.Label(face_detection_frame, text="Real-time Detected Features", background='black', foreground='white')
    features_label.pack(pady=2, padx=2, side=tk.RIGHT)

    # Create a capture button
    capture_button = ttk.Button(face_detection_frame, text="Capture", command=capture_frame, style='Custom.TButton')
    capture_button.pack(pady=2)

    # Create a button to toggle privacy mode
    privacy_button_text = "Privacy Mode"
    privacy_button = ttk.Button(face_detection_frame, text=privacy_button_text, command=toggle_privacy_mode, style='Custom.TButton')
    privacy_button.pack(pady=2)
    
    # Create a back/exit button
    back_button = ttk.Button(face_detection_frame, text="Exit", command=stop_capture, style='Custom.TButton')
    back_button.pack(pady=2)

    # Function to update the features label with real-time detected features
    def update_features_label(face_count, eye_count, eye_colour, skin_colours, hair_colours):
        features_text = f"Faces Detected: {face_count}\nEyes Detected: {eye_count}\nEye Colours: {eye_colour}\nSkin Colours: {skin_colours}\nHair Colours: {hair_colours}"
        features_label.config(text=features_text)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if not privacy_mode_active:

            # Draw rectangles around detected faces code adapted from a solution provided on Stack Overflow: https://stackoverflow.com/questions/75267319/python-opencv-draw-rectangle-from-smaller-frame
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Draw rectangles around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Detect skin color and hair color
                skin_colour = detect_skin_colour(frame[y:y+h, x:x+w])
                hair_colour = detect_hair_colour(frame[y:y+h, x:x+w])

                # Draw rectangles around the eyes
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

                    # Analyse eye color
                    eye_roi = frame[y + ey:y + ey + eh, x + ex:x + ex + ew]
                    eye_colour = detect_eye_colour(eye_roi)
                    
                    # Draw text on the frame indicating the detected eye color
                    eyecolour = str(eye_colour)
                    cv2.putText(frame, eyecolour, (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Update the features label with real-time detected features
                    update_features_label(faces, eyes, eye_colour, skin_colour, hair_colour)

        # Display frame with detected faces and eyes in the GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        label.img = photo
        label.configure(image=photo)
        label.update()

        # Add a small delay to avoid consuming all CPU resources
        root.after(10, None)

    # Release the video capture and writer when exiting the loop
    cap.release()

# HSV colour range identification code adapted from a solution provided on Stack Overflow: https://stackoverflow.com/questions/45513886/opencv-how-can-i-get-the-eye-color
# Function to detect eye colour
def detect_eye_colour(eye_roi):
    # Split eye region into its color channels
    blue_channel = eye_roi[:,:,0]
    green_channel = eye_roi[:,:,1]
    red_channel = eye_roi[:,:,2]
    
    # Calculate mean intensity of each channel
    blue_mean = np.mean(blue_channel)
    green_mean = np.mean(green_channel)
    red_mean = np.mean(red_channel)
    
    # Determine the dominant color based on channel intensities
    if blue_mean > 50 and green_mean < 100 and red_mean < 100:
        eye_color = (255, 0, 0)  
        bestcolour ="Blue"
    elif green_mean > 50 and blue_mean < 100 and red_mean < 100:
        eye_color = (0, 255, 0)  
        bestcolour ="Green"
    elif red_mean > 50 and blue_mean < 100 and green_mean < 100:
        eye_color = (0, 0, 0)  
        bestcolour ="Black"
    else:
        eye_color = (0, 0, 0)  
        bestcolour = "Black"
    
    return bestcolour

# Function to detect skin colour
def detect_skin_colour(face_roi):
    # Convert face region to HSV color space
    skin_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    # Split the HSV channels
    hue_channel = skin_hsv[:,:,0]
    saturation_channel = skin_hsv[:,:,1]
    value_channel = skin_hsv[:,:,2]
    
    # Calculate mean intensity of each channel
    hue_mean = np.mean(hue_channel)
    saturation_mean = np.mean(saturation_channel)
    value_mean = np.mean(value_channel)
    
    # Determine the dominant skin color based on channel intensities
    if hue_mean >= 0 and hue_mean <= 50 and saturation_mean > 25 and value_mean > 50:
        skin_color = (255, 255, 255)  # Light skin 
        bestcolour = "Light"
    else:
        skin_color = (0, 0, 0)  # Dark skin
        bestcolour = "Dark"
    
    return bestcolour


# Function to detect hair colour
def detect_hair_colour(face_roi):
    # Split face region into its color channels
    blue_channel = face_roi[:,:,0]
    green_channel = face_roi[:,:,1]
    red_channel = face_roi[:,:,2]
    
    # Calculate mean intensity of each channel
    blue_mean = np.mean(blue_channel)
    green_mean = np.mean(green_channel)
    red_mean = np.mean(red_channel)
    
    # Determine the dominant hair color based on channel intensities
    if blue_mean < 100 and green_mean < 100 and red_mean < 100:
        hair_color = "Black"
    elif blue_mean > 150 and green_mean > 150 and red_mean > 150:
        hair_color = "Blonde"
    else:
        hair_color = "Black"
    
    return hair_color

# Encryption code adapted from https://blog.bytescrum.com/encrypting-and-decrypting-data-with-fernet-in-python
# Function to save data to the file and encrypt it 
def save_face_image(face_name, face_image, face_data):
    # Generate or load AES key
    key = load_or_generate_key()

    # Convert face image to bytes
    _, image_buffer = cv2.imencode('.png', face_image)
    face_image_bytes = image_buffer.tobytes()

    # Encrypt and save the captured face image to a file
    folder_path = 'captured_faces'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    image_path = os.path.join(folder_path, f'{face_name}.png')

    # Encrypt face image
    cipher = Fernet(key)
    encrypted_image = cipher.encrypt(face_image_bytes)

    with open(image_path, 'wb') as file:
        file.write(encrypted_image)
    print(f"Saved and encrypted image to: {image_path}")

    # Encrypt and save face data to a text file
    face_data_path = os.path.join(folder_path, f'{face_name}.txt')
    encrypted_data = encrypt_data(json.dumps(face_data), key)  # Ensure face_data is JSON serialized
    with open(face_data_path, 'wb') as file:
        file.write(encrypted_data)
    print(f"Saved and encrypted face data to: {face_data_path}")

    # Return the AES key for potential decryption
    return key

# View saved face data
def view_face_data():
    
    # User interface button styling code adapted from a solution on StackOverflow: https://stackoverflow.com/questions/27347981/how-to-change-the-color-of-ttk-button
    # Change colour of button to black
    style = ttk.Style()
    style.theme_use('alt')
    style.configure('Custom.TButton', background='black', foreground='white', borderwidth=1, focusthickness=3, focuscolor='none',width = 20)
    style.map('Custom.TButton', background=[('active','black')])
    
    # Create a new window for the View Face Data page
    view_window = tk.Toplevel(root)
    view_window.title("View Face Data")

    # Create a label to display instructions
    instructions_label = tk.Label(view_window, text="Select a face to view its data:")
    instructions_label.pack(pady=10)

    # Load saved face names from image files
    folder_path = 'captured_faces'
    face_names = [file_name.split('.')[0] for file_name in os.listdir(folder_path) if file_name.endswith('.png')]

    # Select faces
    selected_face = ttk.Combobox(view_window, values=face_names)
    selected_face.pack(pady=2)

    # Function to display detailed information about the selected face
    def detail_view():
        selected_name = selected_face.get()
        if selected_name:
            # Display the image and data with the selected face
            image_path = os.path.join(folder_path, f'{selected_name}.png')
            face_data_path = os.path.join(folder_path, f'{selected_name}.txt')

            try:
                # Load and display image
                with open(image_path, 'rb') as file:
                    encrypted_image = file.read()

                # Decrypt face image
                key = load_or_generate_key()
                face_image = decrypt_face_data(encrypted_image, key)

                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_image_rgb)
                tk_image = ImageTk.PhotoImage(pil_image)

                # Update the face label with the face image
                face_label.config(image=tk_image)
                face_label.image = tk_image

                # Load and display face data
                with open(face_data_path, 'rb') as file:
                    encrypted_data = file.read()

                # Decrypt face data
                decrypted_data = decrypt_data(encrypted_data, key)
                face_data = decrypted_data.decode().splitlines()

                # Update the features label with the selected face's data
                features_text = "\n".join(face_data)
                features_label.config(text=features_text)

            except FileNotFoundError:
                messagebox.showerror("Error", "No face data found.")
        else:
            messagebox.showinfo("Error", "Please select a face.")
    
    # Function to export face data
    def export_data():
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if export_dir:
            selected_name = selected_face.get()
            if selected_name:
                # Load the face image and data
                image_path = os.path.join(folder_path, f'{selected_name}.png')
                face_data_path = os.path.join(folder_path, f'{selected_name}.txt')

                try:
                    # Load the face image
                    with open(image_path, 'rb') as file:
                        encrypted_image = file.read()

                    # Decrypt face image
                    key = load_or_generate_key()
                    face_image = decrypt_face_data(encrypted_image, key)

                    # Load face data
                    with open(face_data_path, 'rb') as file:
                        encrypted_data = file.read()

                    # Decrypt face data
                    decrypted_data = decrypt_data(encrypted_data, key)
                    face_data = json.loads(decrypted_data)

                    # Save the decrypted image and data to the export directory
                    export_image_path = os.path.join(export_dir, f'{selected_name}.png')
                    cv2.imwrite(export_image_path, face_image)

                    export_data_path = os.path.join(export_dir, f'{selected_name}.txt')
                    with open(export_data_path, 'w') as file:
                        json.dump(face_data, file)

                    messagebox.showinfo("Export Successful", "Face data exported successfully.")
                except FileNotFoundError:
                    messagebox.showerror("Error", "No face data found.")
            else:
                messagebox.showinfo("Error", "Please select a face.")
    
    # Create a button to view detailed information about the selected face
    detail_button = ttk.Button(view_window, text="Detail View", command=detail_view, style='Custom.TButton')
    detail_button.pack(pady=2)

    # Create a box to display features
    face_label = ttk.Label(view_window, text="Face image", background='black', foreground='white')
    face_label.pack(pady=2, side=tk.LEFT, padx=(10, 5))

    # Create a box to display features
    features_label = ttk.Label(view_window, text="Features", background='black', foreground='white')
    features_label.pack(pady=2, side=tk.LEFT, padx=(5, 10))

    # Create a button to export face data
    export_button = ttk.Button(view_window, text="Export Data", command=export_data, style='Custom.TButton')
    export_button.pack(pady=2)

    # Create a button to exit and return to the main menu
    exit_button = ttk.Button(view_window, text="Exit", command=view_window.destroy, style='Custom.TButton')
    exit_button.pack(pady=2)

# View saved face data
def delete_face_data():
    
    # User interface button styling code adapted from a solution on StackOverflow: https://stackoverflow.com/questions/27347981/how-to-change-the-color-of-ttk-button
    # Change colour of button to black
    style = ttk.Style()
    style.theme_use('alt')
    style.configure('Custom.TButton', background='black', foreground='white', borderwidth=1, focusthickness=3, focuscolor='none',width = 20)
    style.map('Custom.TButton', background=[('active','black')])
    
    # Create a new window for the View Face Data page
    view_window = tk.Toplevel(root)
    view_window.title("Delete Face Data")

    # Create a label to display instructions
    instructions_label = tk.Label(view_window, text="Select a face to delete its data:")
    instructions_label.pack(pady=10)

    # Load saved face names from image files
    folder_path = 'captured_faces'
    face_names = [file_name.split('.')[0] for file_name in os.listdir(folder_path) if file_name.endswith('.png')]

    # Select faces
    selected_face = ttk.Combobox(view_window, values=face_names)
    selected_face.pack(pady=2)

    # Function to delete face data
    def delete_face():
        selected_name = selected_face.get()
        if selected_name:
            # Delete the face image file if it exists
            image_path = os.path.join('captured_faces', f'{selected_name}.png')
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted image: {image_path}")

            # Delete the face data text file if it exists
            txt_file_path = os.path.join('captured_faces', f'{selected_name}.txt')
            if os.path.exists(txt_file_path):
                os.remove(txt_file_path)
                print(f"Deleted text file: {txt_file_path}")

            messagebox.showinfo("Face Data Deleted", f"Face data for '{selected_name}' has been deleted.")

    # Create a button to delete face data
    delete_button = ttk.Button(view_window, text="Delete Face", command=delete_face, style='Custom.TButton')
    delete_button.pack(pady=2)

    # Create a button to exit and return to the main menu
    exit_button = ttk.Button(view_window, text="Exit", command=view_window.destroy, style='Custom.TButton')
    exit_button.pack(pady=2)

# Warning to user about program before it starts
def warning_popup():
    # Create a warning popup
    response = messagebox.askyesno("Important Ethical Warning", "By continuing, you acknowledge that this program initiates face detection. Please be aware that the use of such technology carries ethical responsibilities. This system is designed for legitimate and ethical purposes only, such as identification or authentication. Misuse, including but not limited to surveillance without consent or discrimination, is strictly prohibited and may violate laws and ethical standards. Ensure that you have appropriate consent and use this tool responsibly. Do you wish to proceed?")
    if response:
        start_face_detection()

def create_main_menu():

    # User interface button styling code adapted from a solution on StackOverflow: https://stackoverflow.com/questions/27347981/how-to-change-the-color-of-ttk-button
    # Change colour of button to black
    style = ttk.Style()
    style.theme_use('alt')
    style.configure('Custom.TButton', background='black', foreground='white', borderwidth=1, focusthickness=3, focuscolor='none',width = 20)
    style.map('Custom.TButton', background=[('active','black')])

    global main_menu_frame

    # Create main menu frame
    main_menu_frame = ttk.Frame(root)
    main_menu_frame.pack()


    display_label = tk.Label(main_menu_frame, text="Face Detection System")
    display_label.pack(pady=2)

    # Create button to start face detection
    start_button = ttk.Button(main_menu_frame, text="Start Face Detection", command=warning_popup, style='Custom.TButton')
    start_button.pack(pady=2)

    # Create button to view face data
    view_button = ttk.Button(main_menu_frame, text="View Face Data", command=lambda: view_face_data(), style='Custom.TButton')
    view_button.pack(pady=2)

    # Create button to delete face data
    delete_button = ttk.Button(main_menu_frame, text="Delete Face Data", command=lambda: delete_face_data(), style='Custom.TButton')
    delete_button.pack(pady=2)

    # Create exit button
    exit_button = ttk.Button(main_menu_frame, text="Exit", command=root.quit, style='Custom.TButton')
    exit_button.pack(pady=2)
    
# Create main application window
root = tk.Tk()
root.title("Face Detection System")

# Call function to create the main menu
create_main_menu()

root.mainloop()
