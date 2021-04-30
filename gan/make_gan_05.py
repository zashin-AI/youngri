# https://rubikscode.net/2018/12/17/implementing-gan-dcgan-with-python/

#######
# DC gan
#######

from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ImageHelper(object):
    def save_image(self, generated, epoch, directory):
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(generated[count, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                count += 1
        fig.savefig("{}/{}.png".format(directory, epoch))
        plt.close()

#--------------------------------------------------------------------
# Implementation

# Keras modules
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class DCGAN():
    def __init__(self, image_shape, generator_input_dim, image_hepler, img_channels):
        optimizer = Adam(0.0002, 0.5)
        
        self._image_helper = image_hepler
        self.img_shape = image_shape
        self.generator_input_dim = generator_input_dim
        self.channels = img_channels

        # Build models
        self._build_generator_model()
        self._build_and_compile_discriminator_model(optimizer)
        self._build_and_compile_gan(optimizer)

    def train(self, epochs, train_data, batch_size):
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):
            #  Train Discriminator
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            genenerated = self._predict_noise(batch_size)
            loss_real = self.discriminator_model.train_on_batch(batch, real)
            loss_fake = self.discriminator_model.train_on_batch(genenerated, fake)
            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)

            #  Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.generator_input_dim))
            generator_loss = self.gan.train_on_batch(noise, real)

            # Plot the progress
            print ("———————————————————")
            print ("******************Epoch {}***************************".format(epoch))
            print ("Discriminator loss: {}".format(discriminator_loss[0]))
            print ("Generator loss: {}".format(generator_loss))
            print ("———————————————————")
            
            history.append({"D":discriminator_loss[0],"G":generator_loss})
            
            # Save images from every hundereth epoch generated images
            if epoch % 100 == 0:
                self._save_images(epoch)
                np.save('C:/nmb/gan/make_gan_05/npy/DC_n22144,e10000,b100_{}.npy'.format(str(epoch).zfill(5)), arr=noise)

        self._plot_loss(history)        # 악 깜빡했다!!
                    
    def _build_generator_model(self):
        generator_input = Input(shape=(self.generator_input_dim,))
        generator_seqence = Sequential(
                [Dense(128 * 7 * 7, activation="relu", input_dim=self.generator_input_dim),
                 Reshape((7, 7, 128)),
                 UpSampling2D(),
                 Conv2D(128, kernel_size=3, padding="same"),
                 BatchNormalization(momentum=0.8),
                 Activation("relu"),
                 UpSampling2D(),
                 Conv2D(64, kernel_size=3, padding="same"),
                 BatchNormalization(momentum=0.8),
                 Activation("relu"),
                 Conv2D(self.channels, kernel_size=3, padding="same"),
                 Activation("tanh")])
    
        generator_output_tensor = generator_seqence(generator_input)       
        self.generator_model = Model(generator_input, generator_output_tensor)
        
    def _build_and_compile_discriminator_model(self, optimizer):
        discriminator_input = Input(shape=self.img_shape)
        discriminator_sequence = Sequential(
                [Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(64, kernel_size=3, strides=2, padding="same"),
                 ZeroPadding2D(padding=((0,1),(0,1))),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(128, kernel_size=3, strides=2, padding="same"),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(256, kernel_size=3, strides=2, padding="same"),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Flatten(),
                 Dense(1, activation='sigmoid')])
    
        discriminator_tensor = discriminator_sequence(discriminator_input)
        self.discriminator_model = Model(discriminator_input, discriminator_tensor)
        self.discriminator_model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator_model.trainable = False
    
    def _build_and_compile_gan(self, optimizer):
        real_input = Input(shape=(self.generator_input_dim,))
        generator_output = self.generator_model(real_input)
        discriminator_output = self.discriminator_model(generator_output)        
        
        self.gan = Model(real_input, discriminator_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def _save_images(self, epoch):
        generated = self._predict_noise(25)
        generated = 0.5 * generated + 0.5
        self._image_helper.save_image(generated, epoch, "C:/nmb/gan/make_gan_05/")
    
    def _predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.generator_input_dim))
        return self.generator_model.predict(noise)
        
    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()

# -------------------------------------------------------
# 훈련

import cv2

X = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/resize_f_mels.npy')
# X_train = X/(np.max(X))
X_train = np.expand_dims(X, axis=3)

image_helper = ImageHelper()
generative_advarsial_network = DCGAN(X_train[0].shape, 10000, image_helper, 1)
generative_advarsial_network.train(10000, X_train, batch_size=100)

# (self, image_shape, generator_input_dim, image_hepler, img_channels):

