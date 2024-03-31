import tkinter as tk
from tkinter import ttk, filedialog, font as tkfont
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize the main window
root = tk.Tk()
root.title("Weather Condition Prediction")
root.geometry("1024x768") 

# Define fonts
default_font = tkfont.nametofont("TkDefaultFont")
default_font.configure(size=16)
big_font = tkfont.Font(family="Helvetica", size=16)


def load_selected_model(model_name):

    # select the model based on given name
    if model_name == 'VGGNet':
        model_path = 'models/model_vggnet.keras'
    elif model_name == 'ResNet':
        model_path = 'models/model_resnet.keras'
    elif model_name == 'Model1':
        model_path = 'models/model_1.keras'
    elif model_name == 'Model2':
        model_path = 'models/model_2.keras'
    elif model_name == 'Model3':
        model_path = 'models/model_3.keras'
    else:
        model_path = 'models/model_1.keras'

    # load the model and return it
    result_text.config(text=f"Loading {model_name}...")
    model = load_model(model_path)
    result_text.config(text="Model loaded, predicting ...")
    return model

# load and preprocess the given image
def preprocess_image(image_path, target_size=(224, 224, 3)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)    
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    print(img_array_expanded_dims.shape)
    return img_array_expanded_dims

# predict the weather based on given image and selected model
def predict(image_path, model_choice):
    model = load_selected_model(model_choice)
    processed_image = preprocess_image(image_path)
    probabilities = model.predict(processed_image)
    return probabilities

# update the bar chart with predicted probabilities
def update_chart(canvas, ax, probabilities):
    probabilities = probabilities.flatten()
    conditions = ['Day', 'Foggy', 'Night', 'Rain', 'Snow']
    predicted_class = np.argmax(probabilities)
    print(f"predicted class: {predicted_class} - probabailities: {probabilities}")
    ax.clear()    
    result_text.config(text=f"Predictied class: {conditions[predicted_class]} ({model_choice_var.get()} model!)")
    ax.bar(conditions, probabilities, color='skyblue')
    ax.set_ylabel('Probability', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    canvas.draw()

# called when "Apply Prediction" button is clicked
def on_predict():
    if not filepath:
        result_text.config(text="Please upload an image first.")
        return
    probabilities = predict(filepath, model_choice_var.get())

    
    result_text.config(text="Prediction applied. See chart for probabilities.")
    update_chart(chart_canvas, chart_ax, probabilities)

# upload an image and display it
def upload_image():
    global filepath
    filepath = filedialog.askopenfilename()
    if filepath:
        display_image(filepath)

# display the uploaded image in the GUI
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img


# --- GUI --- #
# model selection dropdown
model_choice_var = tk.StringVar()
model_dropdown = ttk.Combobox(root, textvariable=model_choice_var, state='readonly', font=big_font)
model_dropdown['values'] = ('VGGNet', 'ResNet', 'Model1', 'Model2', 'Model3')
model_dropdown.pack(pady=20)

# upload picture button
upload_btn = tk.Button(root, text="Upload Picture", command=upload_image, font=big_font)
upload_btn.pack(pady=10)

# image display section
image_label = tk.Label(root)
image_label.pack(pady=10)

# apply prediction button
predict_btn = tk.Button(root, text="Apply Prediction", command=on_predict, font=big_font)
predict_btn.pack(pady=10)

# text field to show the result!
result_text = tk.Label(root, text="Prediction result will be displayed here", font=big_font)
result_text.pack(pady=10)

# probability bar chart section
fig, chart_ax = plt.subplots(figsize=(6, 4))
chart_canvas = FigureCanvasTkAgg(fig, master=root)
chart_canvas.draw()
chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# start the GUI main loop
root.mainloop()
