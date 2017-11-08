
# Rather than generating and saving the new augmented data into the hardisk , we
# we can do the data augemntation on the fly using inbuilt funtionalities in keras
#
# -Using fit_generator instead of the fit, while training the cnn:
# https://keras.io/models/sequential/
#
# -Using the flow Method from the ImageDataGenerator of keras as a generator
# https://keras.io/preprocessing/image/
#
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('test.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
print x.shape
x = x.reshape
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
print x.shape
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
os.makedirs('preview')
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


# # Save augmented images to file
# from keras.datasets import mnist
# from keras.preprocessing.image import ImageDataGenerator
# from matplotlib import pyplot
# import os
# from keras import backend as K
# K.set_image_dim_ordering('th')
# # load data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# # reshape to be [samples][pixels][width][height]
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# # convert from int to float
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# # define data preparation
# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
# # fit parameters from data
# datagen.fit(X_train)
# # configure batch size and retrieve one batch of images
# os.makedirs('images')
# for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='images', save_prefix='aug', save_format='png'):
# 	# create a grid of 3x3 images
# 	for i in range(0, 9):
# 		pyplot.subplot(330 + 1 + i)
# 		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
# 	# show the plot
# 	pyplot.show()
# 	break
