import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
from keras.models import load_model
import numpy as np
 
# load and prepare the image
def load_image(filename):
	# load the image
	test_image = tf.keras.utils.load_img(filename, target_size=(200, 200))
	# convert to array
	test_image = tf.keras.utils.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	return test_image
 
# load an image and predict the class
def classify_image():
 # get the filename of the selected image
	filename = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
 
	# load the selected image
	image = Image.open(filename)
	# resize the image to fit in the GUI
	image = image.resize((400, 400), Image.ANTIALIAS)
	# display the image on the GUI
	photo = ImageTk.PhotoImage(image)
	canvas.itemconfig(image_on_canvas, image=photo)
	canvas.image = photo
	
	# load the image and classify it using the trained model
	img = load_image(filename)
	model = load_model('model_3-4.h5')
	result = model.predict(img)
	
	# display the result on the GUI
	if result >= 0.5:
		label.config(text="It's a dog!",font=("Arial",60))
	else:
		label.config(text="It's a cat!",font=("Arial", 60))
 
# create the GUI
root = tk.Tk()
root.title("Image Classifier")
root.resizable(False, False)
# create the canvas to display the image
canvas = tk.Canvas(root, width=400, height=400)

canvas.pack()
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW)

# create a button to select the image
button = tk.Button(root, text="Select Image", command=classify_image)
button.pack()

# create a label to display the result
label = tk.Label(root, text="")
label.pack()

root.mainloop()