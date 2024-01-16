import time
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn import svm
from os.path import exists
import pickle
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from Pyfhel import Pyfhel

# For testing federated learning, centralized model and so on
# For debugging and preporatory step for implementing
# simple_server and simple_client
# Also used for generating encryption keyes

np.set_printoptions(precision=15)

HE = Pyfhel()

ckks_params = {
    'scheme': 'CKKS',
    'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
                        #  encoded in a single ciphertext. 
                        #  Typ. 2^D for D in [10, 16]
    'scale': 2**30,     # All the encodings will use it for float->fixed point
                        #  conversion: x_fix = round(x_float * scale)
                        #  You can use this as default scale or use a different
                        #  scale on each operation (set in HE.encryptFrac)
    'qi': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain. 
                        # Intermediate values should be  close to log2(scale)
                        # for each operation, to have small rounding errors.
}

HE.contextGen(**ckks_params)  # Generate context for bfv scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys

# store HE parameters and keys for client use
# dir_name = "HE_instance"
# HE.save_context(dir_name + "/context")
# HE.save_public_key(dir_name + "/pub.key")
# HE.save_secret_key(dir_name + "/sec.key")

# use homomorphic encryption
use_encryption = True

# compute svm for comparison
do_svm = False

# check saved model
verify_model = False

# train centralized model
do_centralized = False

# plot points
plot_points = False

# raw centralized training for reference
do_centralized_raw = False

# federated learning
do_federated = True

# train and test data, two clients
local_data = pickle.load(open("local_data.p", 'rb'))

X_train5, y_train5, X_test5, y_test5 = local_data[5]
X_train7, y_train7, X_test7, y_test7 = local_data[7]

X_train = np.concatenate([X_train5, X_train7], axis=0)
X_test = np.concatenate([X_test5, X_test7], axis=0)
y_train = np.concatenate([y_train5, y_train7])
y_test = np.concatenate([y_test5, y_test7])
X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test])

# plotting
colors = ['g' if c == 0 else 'r' for c in y]
if plot_points:
    plt.scatter(*zip(*X), c=colors, s=0.5)
    plt.show()

# centralized training, svm
if do_svm:
    # linear kernel
    # clf = svm.SVC(kernel='linear')
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    acc_train = (pred_train == y_train).astype(int).sum()/y_train.shape[0]

    pred_test = clf.predict(X_test)
    acc_test = (pred_test == y_test).astype(int).sum()/y_test.shape[0]

    print("acc_train:", acc_train)
    print("acc_test:", acc_test)

# centralized training, multi layer perceptron
if do_centralized:
    metrics = ['accuracy']
    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=0.01)
    model.compile(loss='mse', optimizer=optimizer, metrics=metrics)
    model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=200, batch_size=None)
    # model.save('./saved_models/my_model')

# checking saved model
if verify_model:
    model = keras.models.load_model('./saved_models/my_model')
    preds = np.round(model(X_test))
    preds = preds.reshape(preds.shape[0])
    print("Population acc:", (preds == y_test).astype(int).sum()/preds.shape[0])
    preds_p = np.round(model(Xp_test))
    preds_p = preds_p.reshape(preds_p.shape[0])
    print("Individual acc %s:" % person, (preds_p == yp_test).astype(int).sum()/preds_p.shape[0])

# centralized training multilayer perceptron, raw gradient descent
if do_centralized_raw:
    metrics = ['accuracy']
    global_epochs = 400
    # local_epochs = 1
    model_raw = keras.Sequential()
    model_raw.add(keras.layers.Dense(10, activation='relu'))
    model_raw.add(keras.layers.Dense(10, activation='relu'))
    model_raw.add(keras.layers.Dropout(0.5))
    model_raw.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.01)
    # learning_rate = 0.01

    mse_loss = tf.keras.losses.MeanSquaredError()

    y_train_reshape = y_train.reshape([y_train.shape[0], 1])
    y_test_reshape = y_test.reshape([y_test.shape[0], 1])
    for epoch in range(global_epochs):
    
        with tf.GradientTape() as t:
            loss = mse_loss(y_train_reshape, model_raw(X_train))

            params = model_raw.trainable_variables

            grads = t.gradient(loss, params)

            #for p, gr in zip(model_raw.trainable_variables, grads):
            #    p.assign_sub(learning_rate*gr)

            # gradient step
            optimizer.apply_gradients(zip(grads, model_raw.trainable_weights))

            tr_loss = mse_loss(y_train_reshape, model_raw(X_train)).numpy()
            test_loss = mse_loss(y_test_reshape, model_raw(X_test)).numpy()

            pred_train = np.round(model_raw(X_train).numpy())
            acc_train = (pred_train == y_train_reshape).astype(int).sum()/y_train_reshape.shape[0]

            pred_test = np.round(model_raw(X_test).numpy())
            acc_test = (pred_test == y_test_reshape).astype(int).sum()/y_test_reshape.shape[0]

            print("epoch:", epoch, "training loss:", tr_loss, "val loss:", test_loss,
                  "acc train:", acc_train, "acc test", acc_test)


if not do_federated:
    sys.exit()


# federated learning
global_epochs = 100
local_epochs = 1

participants = local_data.keys()

metrics = ['accuracy']

# global model used
model_fed = keras.Sequential()
model_fed.add(keras.layers.Dense(10, activation='relu'))
model_fed.add(keras.layers.Dense(10, activation='relu'))
model_fed.add(keras.layers.Dropout(0.5))
model_fed.add(keras.layers.Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=0.01)
model_fed.compile(loss='mse', optimizer=optimizer, metrics=metrics)
model_fed.build([1,2])
learning_rate = 0.01

# loss function
mse_loss = tf.keras.losses.MeanSquaredError()

y_train_reshape = y_train.reshape([y_train.shape[0], 1])
y_test_reshape = y_test.reshape([y_test.shape[0], 1])

num_data_points = {k: local_data[k][0].shape[0] for k in local_data}

num_data_total = sum([num_data_points[k] for k in num_data_points])

for epoch in range(global_epochs):

    global_params = []
    global_params_enc = []
    
    for part in participants:

        # define local model
        model_loc = keras.Sequential()
        model_loc.add(keras.layers.Dense(10, activation='relu'))
        model_loc.add(keras.layers.Dense(10, activation='relu'))
        model_loc.add(keras.layers.Dropout(0.5))
        model_loc.add(keras.layers.Dense(1, activation='sigmoid'))
        optimizer_loc = Adam(learning_rate=0.01)
        model_loc.compile(loss='mse', optimizer=optimizer_loc, metrics=metrics)
        model_loc.build([1,2])

        # update local model
        for p_l, p_g in zip(model_loc.trainable_variables, model_fed.trainable_variables):
            p_l.assign(p_g.numpy())

        # local training data
        X_train_loc = local_data[part][0]
        y_train_loc = local_data[part][1]
        y_train_loc_reshape = y_train_loc.reshape([y_train_loc.shape[0], 1])

        X_test_loc = local_data[part][2]
        y_test_loc = local_data[part][3]
        y_test_loc_reshape = y_test_loc.reshape([y_test_loc.shape[0], 1])

        for l_epoch in range(local_epochs):

            with tf.GradientTape() as t:

                loss = mse_loss(y_train_loc_reshape, model_loc(X_train_loc))

                params = model_loc.trainable_variables

                grads = t.gradient(loss, params)

                #for p, gr in zip(model_loc.trainable_variables, grads):
                #    p.assign_sub(learning_rate*gr)

                # local gradient step
                optimizer_loc.apply_gradients(zip(grads, model_loc.trainable_weights))

                tr_loss = mse_loss(y_train_loc_reshape, model_loc(X_train_loc)).numpy()
                test_loss = mse_loss(y_test_loc_reshape, model_loc(X_test_loc)).numpy()

                pred_train = np.round(model_loc(X_train_loc).numpy())
                acc_train = (pred_train == y_train_loc_reshape).astype(int).sum()/y_train_loc_reshape.shape[0]

                pred_test = np.round(model_loc(X_test_loc).numpy())
                acc_test = (pred_test == y_test_loc_reshape).astype(int).sum()/y_test_loc_reshape.shape[0]

                print("epoch:", epoch, "local epoch", l_epoch, "part:", part, "local training loss:", tr_loss, "local test loss:", test_loss, "local acc train:", acc_train, "local acc test", acc_test)

        # add weighted local parameters to global parameters
        for ii, p_l in enumerate(model_loc.trainable_variables):
            if not use_encryption:
                if len(global_params) < ii+1:
                    global_params.append(0)
                global_params[ii] += (num_data_points[part]/num_data_total)*p_l.numpy()
            else:
                if len(global_params_enc) < ii+1:
                    zeros_enc = HE.encrypt(np.zeros(p_l.numpy().flatten().shape))
                    global_params_enc.append(zeros_enc)
                update_enc = HE.encrypt((num_data_points[part]/num_data_total)*p_l.numpy().flatten())
                global_params_enc[ii] += update_enc

    # update global parameters
    for ii, p_g in enumerate(model_fed.trainable_variables):
        if not use_encryption:
            p_g.assign(global_params[ii])
        else:
            global_decrypt = HE.decryptFrac(global_params_enc[ii])
            global_decrypt = global_decrypt[:p_g.numpy().size]
            p_g.assign(global_decrypt.reshape(p_g.numpy().shape))

    print()
    tr_loss = mse_loss(y_train_reshape, model_fed(X_train)).numpy()
    test_loss = mse_loss(y_test_reshape, model_fed(X_test)).numpy()
    pred_train = np.round(model_fed(X_train).numpy())
    acc_train = (pred_train == y_train_reshape).astype(int).sum()/y_train_reshape.shape[0]
    pred_test = np.round(model_fed(X_test).numpy())
    acc_test = (pred_test == y_test_reshape).astype(int).sum()/y_test_reshape.shape[0]
    print("epoch:", epoch, "global training loss:", tr_loss, "global test loss", test_loss, "global acc train:", acc_train, "global test acc", acc_test)
    print()
