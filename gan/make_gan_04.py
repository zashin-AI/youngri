<<<<<<< HEAD
# https://rubikscode.net/2018/12/17/implementing-gan-dcgan-with-python/

#######
# 기본 간
#######

from __future__ import print_function, division

import os
import numpy as np
import imageio
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

# from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keras modules
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

class GAN():
    def __init__(self, image_shape, generator_input_dim, image_hepler):
        optimizer = Adam(0.0002, 0.5)
        
        self._image_helper = image_hepler
        self.img_shape = image_shape
        self.generator_input_dim = generator_input_dim

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
                np.save('C:/nmb/gan/make_gan_04/npy/n22144, e3000,b100_{}.npy'.format(str(epoch).zfill(5)), arr=noise)
                
        self._plot_loss(history)
        # self._image_helper.makegif("C:/nmb/gan/make_gan_04/gif/")        
    
    def _build_generator_model(self):
        generator_input = Input(shape=(self.generator_input_dim,))
        generator_seqence = Sequential(
                [Dense(256, input_dim=self.generator_input_dim),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(512),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(1024),
                 LeakyReLU(alpha=0.2),
                 BatchNormalization(momentum=0.8),
                 Dense(np.prod(self.img_shape), activation='tanh'),
                 Reshape(self.img_shape)])
    
        generator_output_tensor = generator_seqence(generator_input)       
        self.generator_model = Model(generator_input, generator_output_tensor)
        
    def _build_and_compile_discriminator_model(self, optimizer):
        discriminator_input = Input(shape=self.img_shape)
        discriminator_sequence = Sequential(
                [Flatten(input_shape=self.img_shape),
                 Dense(512),
                 LeakyReLU(alpha=0.2),
                 Dense(256),
                 LeakyReLU(alpha=0.2),
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
        self._image_helper.save_image(generated, epoch, "C:/nmb/gan/make_gan_04/")
    
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

# ----------------------------------------------------
# 훈련

X = np.load('C:/nmb/nmb_data/0418_balance_denoise_npy/denoise_balance_f_mels.npy')
# X_train = X/127.5 – 1.
X_train = X
X_train = np.expand_dims(X_train, axis=3)

image_helper = ImageHelper()
generative_advarsial_network = GAN(X_train[0].shape, 22144, image_helper)
# (self, image_shape, generator_input_dim, image_hepler)
generative_advarsial_network.train(3000, X_train, batch_size=100)
=======
# gan 직접 만들어보기
# https://github.com/golbin/TensorFlow-Tutorials/blob/master/09%20-%20GAN/01%20-%20GAN.py
# 튜닝중

# 내 목소리 1000개로 학습시키기

# 1점대로 돌리기
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

me_ds1 = np.load('C:/nmb/nmb_data/npy/me_thousand.npy')
me_ds2 = np.load('C:/nmb/nmb_data/npy/me_thousand.npy')
me_ds = np.concatenate([me_ds1, me_ds2],0)

fm_lb = np.load('C:/nmb/nmb_data/npy/1s_2m_total_fm_label.npy')
f_lb = fm_lb[:2000]

me_ds = me_ds.reshape(me_ds.shape[0], me_ds.shape[1]*me_ds.shape[2])
print(me_ds.shape)
print(f_lb.shape)
# (1000, 22144)
# (1000,)

# ---------------------------------------------------------------------------
# 옵션 설정

total_epoch = 10000
# batch_size = 100
batch_size = 2000
learning_rate = 0.0002
# 신경망 레이어 구성 옵션
n_hidden = 3000
n_input = 128*173
n_noise = 22144  # 생성기의 입력값으로 사용할 노이즈의 크기

# ---------------------------------------------------------------------------
# 신경망 모델 구성

# GAN 도 Unsupervised 학습이므로 Autoencoder 처럼 Y 를 사용하지 않습니다.
X = tf.placeholder(tf.float32, [None, n_input])
# 노이즈 Z를 입력값으로 사용합니다.
Z = tf.placeholder(tf.float32, [None, n_noise])
# placeholder(dtype: 데이터 타입, shape=None:쉐잎, name=None:이름 지정)

# 생성기 신경망에 사용하는 변수들입니다.
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 판별기 신경망에 사용하는 변수들입니다.
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
# 판별기의 최종 결과값은 얼마나 진짜와 가깝냐를 판단하는 한 개의 스칼라값입니다.
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

# 생성기(G) 신경망을 구성합니다.
def generator(noise_z):
    hidden = tf.nn.relu(
                    tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(
                    tf.matmul(hidden, G_W2) + G_b2)

    return output

# 판별기(D) 신경망을 구성합니다.
def discriminator(inputs):
    hidden = tf.nn.relu(
                    tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(
                    tf.matmul(hidden, D_W2) + D_b2)

    return output


# 랜덤한 노이즈(Z)를 만듭니다.
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))


# 노이즈를 이용해 랜덤한 이미지를 생성합니다.
G = generator(Z)
# 노이즈를 이용해 생성한 이미지가 진짜 이미지인지 판별한 값을 구합니다.
D_gene = discriminator(G)
# 진짜 이미지를 이용해 판별한 값을 구합니다.
D_real = discriminator(X)

# 논문에 따르면, GAN 모델의 최적화는 loss_G 와 loss_D 를 최대화 하는 것 입니다.
# 다만 loss_D와 loss_G는 서로 연관관계가 있기 때문에 두 개의 손실값이 항상 같이 증가하는 경향을 보이지는 않을 것 입니다.
# loss_D가 증가하려면 loss_G는 하락해야하고, loss_G가 증가하려면 loss_D는 하락해야하는 경쟁관계에 있기 때문입니다.
# 논문의 수식에 따른 다음 로직을 보면 loss_D 를 최대화하기 위해서는 D_gene 값을 최소화하게 됩니다.
# 판별기에 진짜 이미지를 넣었을 때에도 최대값을 : tf.log(D_real)
# 가짜 이미지를 넣었을 때에도 최대값을 : tf.log(1 - D_gene)
# 갖도록 학습시키기 때문입니다.
# 이것은 판별기는 생성기가 만들어낸 이미지가 가짜라고 판단하도록 판별기 신경망을 학습시킵니다.
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
# 반면 loss_G 를 최대화하기 위해서는 D_gene 값을 최대화하게 되는데,
# 이것은 가짜 이미지를 넣었을 때, 판별기가 최대한 실제 이미지라고 판단하도록 생성기 신경망을 학습시킵니다.
# 논문에서는 loss_D 와 같은 수식으로 최소화 하는 생성기를 찾지만,
# 결국 D_gene 값을 최대화하는 것이므로 다음과 같이 사용할 수 있습니다.
loss_G = tf.reduce_mean(tf.log(D_gene))

# loss_D 를 구할 때는 판별기 신경망에 사용되는 변수만 사용하고,
# loss_G 를 구할 때는 생성기 신경망에 사용되는 변수만 사용하여 최적화를 합니다.
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# GAN 논문의 수식에 따르면 loss 를 극대화 해야하지만, minimize 하는 최적화 함수를 사용하기 때문에
# 최적화 하려는 loss_D 와 loss_G 에 음수 부호를 붙여줍니다.
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                                                         var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G,
                                                         var_list=G_var_list)

                                                         
# ---------------------------------------------------------------------------
# 신경망 모델 학습

# next_batch 정의
def next_batch(num, data, labels):
    
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    labels_shuffle = np.asarray(labels_shuffle.reshape(len(labels_shuffle), 1))

    return data_shuffle, labels_shuffle


sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(2000/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(num = batch_size, data = me_ds, labels = f_lb)
        noise = get_noise(batch_size, n_noise)

        # 판별기와 생성기 신경망을 각각 학습시킵니다.
        _, loss_val_D = sess.run([train_D, loss_D],
                                 feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G],
                                 feed_dict={Z: noise})

    print('Epoch:', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))

    # ---------------------------------------------------------------------------
    # 학습이 되어가는 모습을 보기 위해 주기적으로 이미지를 생성하여 저장

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        filename = 'me_bach2000_noise_22144_nhidden3000_epoch10000'
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (128, 173)))

        plt.savefig('C:/nmb/gan/sample/'+ filename + '_{}.png'.format(str(epoch).zfill(5)), bbox_inches='tight')
        plt.close(fig)

        np.save('C:/nmb/gan/npy/'+ filename + '_total{}.npy'.format(str(epoch).zfill(5)), arr=samples)

print('최적화 완료!')

# --------------------------------------------------------------
# Epoch: 0099 D loss: -2.307e-05 G loss: -10.68
>>>>>>> 6d7865c51ce96e775bdf734857f107e9bb0b0f75
