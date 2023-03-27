from keras.utils import load_img, img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(200, 200))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 200, 200, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# load an image and predict the class
def run_example():
    # load the image
    img = load_image('sample_image2.jpg')
    # load model
    model = load_model('files/models/2023-03-19_23-06-01_model_3-4_it_1.h5')
    # predict the class
    result = model.predict(img)

    print(result[0])

# entry point, run the example
run_example()