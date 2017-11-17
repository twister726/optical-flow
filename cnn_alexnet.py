from __future__ import print_function
import keras
# from theano import tensor as T
# from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, normalization, merge, Activation, \
                        Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras import backend as K
from keras import initializers
from keras.layers.core import Lambda, Reshape
from keras.callbacks import ModelCheckpoint
from customlayers import splittensor, LRN2D
import shutil
import os
from load_dataset import load_dataset

try:
    shutil.rmtree('weights', True)
    os.makedirs('weights')
except:
    print('Could not create weights directory')
    pass

def AlexNet():
    inputs = Input(shape=(3, 200, 200))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = LRN2D(name='convpool_1')(conv_1)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', init='glorot_uniform', bias=True, name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = LRN2D()(conv_2)
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', init='glorot_uniform', bias=True, name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', bias=True, init='glorot_uniform', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', init='glorot_uniform', bias=True, name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(16000, name='dense_3')(dense_3)
    reshape = Reshape((400, 40))(dense_3)
    prediction = Activation('softmax', name='softmax')(reshape)

    model = Model(input=inputs, output=prediction)

    return model

batch_size = 32
num_classes = 40*20*20
epochs = 12

img_rows, img_cols = 200, 200

# x_train = pickle.load( open( "train_imgs", "rb" ) )
# x_test = pickle.load( open( "train_labels", "rb" ) )
# y_train = pickle.load( open( "test_imgs", "rb" ) )
# y_test = pickle.load( open( "test_labels", "rb" ) )
#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)


# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = AlexNet()

# model = Sequential()
# # conv1, relu1
# model.add(Conv2D(96, kernel_size=(11, 11),
#                  strides=(4, 4),
#                  activation='relu',
# 				 kernel_initializer='glorot_normal',
# 				 bias_initializer='zeros',
#                  input_shape=input_shape))
# # norm1
# model.add(normalization.LRN2D())
# # pool1
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# # conv2, relu2
# model.add(ZeroPadding2D(padding=2))
# model.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='glorot_normal',
# 								bias_initializer=initializers.constant(value=0.1)))
#
# conv_2 = merge([
# 	    Convolution2D(128,5,5,activation="relu",init='he_normal', name='conv_2_'+str(i+1))(
# 		splittensor(ratio_split=2,id_split=i)(conv_2)
# 	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")
# # norm2
# model.add(normalization.LRN2D())
# # pool2
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# # conv3, relu3
# model.add(ZeroPadding2D(padding=1))
# model.add(Conv2D(384, (3, 3), activation='relu', kernel_initializer='glorot_normal',
# 								bias_initializer=initializers.constant(value=0.0)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
train_set = load_dataset('datasets/Sintel/training', batch_size)
val_set = load_dataset('datasets/Sintel/training', batch_size, wantVal=True)

# for i in range(10):
#     print(next(train_set))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='weights/weights.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1)

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
model.fit_generator(train_set, epochs=100, steps_per_epoch=500,
                    validation_data=val_set, validation_steps=100,
                    callbacks=[checkpointer])
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
