from keras.models import load_model
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# loading the model
model_number = input("Which model would you like to use? Enter a number between 1 and 5. ")

if (model_number != 1):
    model_number = '1'

model = load_model('./models/model_' + model_number + '.h5')

# choosing how many test images will be loaded
iterations = 1
if (len(sys.argv) > 1):
    iterations = int(sys.argv[1])

# processing test images
for i in range(iterations):

    # selecting a random image from the test dataset
    count = random.randint(1, 2500)

    # Loading a random input image from the test dataset
    test_image = load_img('./test/' + str(count) + '.jpg',target_size=(200,200))
    
    # to print the image later
    plt.imshow(test_image)

    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    
    # result array
    result = model.predict(test_image)
    
    # mapping result array with the main name list
    if(result>=0.5):
        print("Dog")
        plt.title('Dog')
    else:
        print("Cat")
        plt.title('Cat')

    # printing the image
    plt.show()