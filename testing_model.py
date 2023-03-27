from keras.models import load_model
from keras.utils import load_img, img_to_array, plot_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('files/models/2023-03-19_23-06-01_model_3-4_it_1.h5')

test_image = load_img('./sample_image.jpg',target_size=(200,200))
    
plt.imshow(test_image)

test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
    
result = model.predict(test_image)

# visualise model
plot_model(
    model,
    to_file='model_3-4.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)
    
if(result>=0.5):
    print(str(result))
    print("Dog")
    plt.title('Dog')
else:
    print(str(result))
    print("Cat")
    plt.title('Cat')

plt.show()

# loading the model
# model_number = input("Which model would you like to use? Enter a number between 1 and 5. ")

# if (model_number != 1):
#     model_number = '1'

#model = load_model('./models/model_' + model_number + '.h5')

# choosing how many test images will be loaded
# iterations = 1
# if (len(sys.argv) > 1):
#     iterations = int(sys.argv[1])

# # processing test images
# for i in range(iterations):

#     # selecting a random image from the test dataset
#     count = random.randint(1, 2500)

#     # Loading a random input image from the test dataset
#     test_image = load_img('./test/' + str(count) + '.jpg',target_size=(200,200))
    
#     # to print the image later
#     plt.imshow(test_image)

#     test_image = img_to_array(test_image)
#     test_image = np.expand_dims(test_image,axis=0)
    
#     # result array
#     result = model.predict(test_image)
    
#     # mapping result array with the main name list
#     if(result>=0.5):
#         print("Dog")
#         plt.title('Dog')
#     else:
#         print("Cat")
#         plt.title('Cat')

#     # printing the image
#     plt.show()