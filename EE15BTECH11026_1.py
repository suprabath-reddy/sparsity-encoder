#use python 3.5
#make sure tensorflow is running in the background

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#resizing the image
x_train = np.reshape(x_train, newshape=(*x_train.shape, 1))
x_train = tf.image.resize_images(images=x_train, size=(14,14))
x = tf.Session().run(x_train)
x_train = np.asarray(x, dtype=np.uint8).reshape(x_train.shape[0], 196) / 255.0

x_test = np.reshape(x_test, newshape=(*x_test.shape, 1))
x_test = tf.image.resize_images(images=x_test, size=(14,14))
x = tf.Session().run(x_test)
x_test = np.asarray(x, dtype=np.uint8).reshape(x_test.shape[0], 196) / 255.0

def show_digits(x, y): 
    x, y = np.reshape(x, (14,14)), np.reshape(y, (14,14))
    fig = plt.figure(figsize=(4, 10))
    fig.add_subplot(1,2,1)
    plt.imshow(x, cmap='gray')
    fig.add_subplot(1,2,2)
    plt.imshow(y, cmap='gray')    
    plt.show()

class network(object):
    def __init__(self, input_dim, hidden_dim, learn_rate, sparsity, regularization):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.learn_rate = learn_rate
        self.s = sparsity
        self.Lambda = regularization
        # intialize weights
        self.A = np.random.normal(0,  1, (self.hidden_dim, self.input_dim))
        self.B = np.random.normal(0,  1, (self.output_dim, self.hidden_dim))
        self.a0 = np.random.normal(0, 1, self.hidden_dim)
        self.b0 = np.random.normal(0, 1, self.output_dim)

    def sigmoid(self, t):
        return 1/(1  + np.exp(-t))

    def dsigmoid(self, t):
        sigt = self.sigmoid(t)
        return sigt*(1-sigt)

    def hidden_layer(self, x):
        z = self.sigmoid(np.dot(x, self.A.T) + self.a0) 
        return z

    def forward_pass(self, x):
        y_hat = self.sigmoid(np.dot(self.hidden_layer(x), self.B.T) + self.b0)
        return y_hat

    def back_propogate(self, X):
        dSSE_A, dSSE_a0 = np.zeros_like(self.A), np.zeros_like(self.a0)
        dSSE_B, dSSE_b0 = np.zeros_like(self.B), np.zeros_like(self.b0)
        Z = self.hidden_layer(X) 
        dZ = Z * (1-Z) 

        Y_out = self.forward_pass(X)
        y_delta = 2*(Y_out-X) * self.dsigmoid(self.b0 + np.dot(Z, self.B.T))

        z_delta = -dZ * np.matmul(y_delta, self.B) 
        zm = np.mean(Z, axis=0)
         
        dKL = self.Lambda * ((-self.s/zm) + ((1-self.s)/(1-zm))) * dZ 
        dSSE_A = np.matmul((z_delta + dKL).T , X)
       
        dSSE_a0 = np.sum((z_delta+dKL), axis=0)
        
        dSSE_B = np.matmul(y_delta.T, Z)
        dSSE_b0 = np.sum(y_delta, axis=0)
 
        A_new = self.A - (self.learn_rate*dSSE_A)
        a0_new = self.a0 - (self.learn_rate*dSSE_a0)
        B_new = self.B - (self.learn_rate*dSSE_B)
        b0_new = self.b0 - (self.learn_rate*dSSE_b0)
        return [A_new, a0_new, B_new, b0_new]

    def loss(self, X):
        y_hat = self.forward_pass(X)
        err = np.sum((X - y_hat)**2)
        zm = np.mean(self.hidden_layer(X), axis=0)
        regularizer = self.Lambda*np.sum(self.s*np.log(self.s/zm) + (1-self.s)*np.log((1-self.s)/(1-zm)))
        return err+regularizer
    
    def train(self, x_train, epochs, batch_size, shuffle=True): 
        epoch = 1
        N = len(x_train)
        while(epoch <= epochs):
            if shuffle:
                indices = np.arange(N)
                np.random.shuffle(indices)
                x_train = x_train[indices]
            
            for batch in np.arange(0, N, batch_size):
                X = x_train[batch:batch+batch_size]
                Y_hat = self.forward_pass(X)
                [self.A, self.a0, self.B, self.b0] = self.back_propogate(X)
            
            print('Epoch:', epoch, ' Loss:', self.loss(x_train))
            
            epoch += 1
        print('Done Training')

#trying for different learn rate and sparsity
SparseAE1 = network(input_dim=196, hidden_dim=225, learn_rate=1e-4, sparsity=0.1, regularization=1)
SparseAE1.train(x_train=x_train, epochs=150, batch_size=300, shuffle=True)

for i in range(5):
    test_out = SparseAE1.forward_pass(x_test[i])
    test_rep = SparseAE1.hidden_layer(x_test[i])
    show_digits(x_test[i], test_out)
    print(test_rep.mean())

SparseAE2 = network(input_dim=196, hidden_dim=256, learn_rate=2e-4, sparsity=0.05, regularization=2)
SparseAE2.train(x_train=x_train, epochs=150, batch_size=300, shuffle=True)

for i in range(5):
    test_out = SparseAE2.forward_pass(x_test[i])
    test_rep = SparseAE2.hidden_layer(x_test[i])
    show_digits(x_test[i], test_out)
    print(test_rep.mean()) 
