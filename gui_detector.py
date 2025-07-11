import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import sys
import os
from tensorflow.keras.models import load_model

def resource_path(relative_path):
    # Works for both dev and .exe
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

model = load_model(resource_path("ai_detector_cnn.h5"))

# Create the GUI window
app = tk.Tk()
app.title("AI Image Detector")
app.geometry("400x500")

# UI elements
label_image = tk.Label(app)
label_result = tk.Label(app, text="", font=("Arial", 14), pady=10)

def predict_image(img_path):
    try:
        # Load and preprocess the image
        img = keras_image.load_img(img_path, target_size=(64, 64))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        print("Raw prediction:", prediction)

        prediction_value = prediction[0][0]  # Safe extraction

        # Classify based on threshold
        if prediction_value > 0.5:
            return f"Real — Confidence: {round(prediction_value, 2)}"
        else:
            return f"Fake (AI-generated) — Confidence: {round(1 - prediction_value, 2)}"

    except Exception as e:
        return f"Error: {str(e)}"

def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")]
    )
    if not file_path:
        return

    try:
        # Display the image
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        label_image.configure(image=img_tk)
        label_image.image = img_tk

        # Show prediction result
        result = predict_image(file_path)
        label_result.config(text=result)

    except Exception as e:
        label_result.config(text=f"Could not open image: {str(e)}")

# Button and layout
btn_browse = tk.Button(app, text="Browse Image", command=browse_image)
btn_browse.pack(pady=20)
label_image.pack()
label_result.pack()

# Start the GUI loop
app.mainloop()
