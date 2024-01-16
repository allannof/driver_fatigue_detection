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
from extract_features import get_events, contract

np.set_printoptions(precision=15)

# HE = Pyfhel()

# ckks_params = {
#     'scheme': 'CKKS',
#     'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
#                         #  encoded in a single ciphertext. 
#                         #  Typ. 2^D for D in [10, 16]
#     'scale': 2**30,     # All the encodings will use it for float->fixed point
#                         #  conversion: x_fix = round(x_float * scale)
#                         #  You can use this as default scale or use a different
#                         #  scale on each operation (set in HE.encryptFrac)
#     'qi': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain. 
#                         # Intermediate values should be  close to log2(scale)
#                         # for each operation, to have small rounding errors.
# }

# HE.contextGen(**ckks_params)  # Generate context for bfv scheme
# HE.keyGen()             # Key Generation: generates a pair of public/secret keys

# use homomorphic encryption
# use_encryption = False

# compute svm for comparison
do_svm = False

# check saved model
verify_model = False

# train centralized model
do_centralized = True

# plot points
plot_points = True

# raw centralized training for reference
do_centralized_raw = False

# store local data
dump_local_data = False

def contraction_mean(z):
    contr = contract(get_events(z))
    if len(contr) > 0:
        return np.array(contr).mean()
    else:
        return None

def events_mean(z):
    return np.array(get_events(z)).mean()

def get_dataframe(datapath, train_window=2400):
    """ Load dataframe from multiple files."""
    first = True
    for filename in sorted(os.listdir(datapath)):
        fullpath = datapath + "/" + filename
        dftmp = pd.read_csv(fullpath, sep=',')
        dftmp = dftmp[['Y', 'Participant', 'EAR', 'Frame']]
        
        # for now
        dftmp = dftmp[dftmp.EAR != 'none']
        dftmp.loc[dftmp.EAR == 'none', 'EAR'] = 'NaN'
        dftmp = dftmp.astype(np.float64)


        EMs = []
        CMs = []
        # step = train_window
        step = 100
        # for ii in range(len(dftmp)//train_window + 1):
        for ii in range(len(dftmp)//step):
            a = ii*step
            b = ii*step+train_window
            em = events_mean(dftmp['EAR'][a:b])
            EMs.append(em)
            cm = contraction_mean(dftmp['EAR'][a:b])
            CMs.append(cm)
    
        y = dftmp["Y"].values[0]
        p = dftmp["Participant"].values[0]
        Ys = [y]*len(EMs)
        Ps = [p]*len(EMs)
        dict_ = {'Y': Ys, 'Participant': Ps, 'EM': EMs, 'CM': CMs}
        dftmp2 = pd.DataFrame(dict_)
        
        if first:
            df = dftmp2
        else:
            df = pd.concat([df, dftmp2], ignore_index=True)
        first = False

    df = df[df.Y != 5.0]
    df.loc[df.Y == 0.0, "Y"] = 0
    df.loc[df.Y == 10.0, "Y"] = 1

    return df
    
def normalize(df, feat='EM', segment=-1):
    """ Normalize data."""
    for p in sorted(list(set(df.Participant.values))):
        val0 = df.loc[(df.Participant == p) & (df.Y == 0), feat][:segment].mean()
        val1 = df.loc[(df.Participant == p) & (df.Y == 1), feat][:segment].mean()
        orig = df.loc[(df.Participant == p),feat]
        df.loc[df.Participant == p,feat+'0'] = orig - val0
        df.loc[df.Participant == p,feat+'1'] = orig - val1
    return df

datapath = '/home/david/gitlab/dais_demo_7_3/feature_extraction/Advanced_Drowsiness_Detection/generated_data'

df = get_dataframe(datapath)
df  = normalize(df, segment=1)
# df  = normalize(df, feat='CM', segment=-1)

# narrow down dataset if desirable
# selection = [5.0, 7.0]
# df = df[df.Participant.isin(selection)]

X = df[['EM0', 'EM1']].values
y = df[['Y']].values

colors = ['g' if c == 0 else 'r' for c in y]

if plot_points:
    plt.scatter(*zip(*X), c=colors, s=0.5)
    plt.show()

# proportion of data used for training
prop = 0.8
index = np.random.choice(range(X.shape[0]), int(np.round(prop*X.shape[0])), replace=False)
rest = [c for c in range(X.shape[0]) if not (c in index)]

X_train = X[index,:]
X_test = X[rest,:]
y_train = y[index].reshape(len(index))
y_test = y[rest].reshape(len(rest))

person = 5.0
Xp_test = df.loc[df['Participant'] == person, ['EM0', 'EM1']].values
yp_test = df.loc[df['Participant'] == person, 'Y'].values

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

if verify_model:
    model = keras.models.load_model('./saved_models/my_model')
    preds = np.round(model(X_test))
    preds = preds.reshape(preds.shape[0])
    print("Population acc:", (preds == y_test).astype(int).sum()/preds.shape[0])
    preds_p = np.round(model(Xp_test))
    preds_p = preds_p.reshape(preds_p.shape[0])
    print("Individual acc %s:" % person, (preds_p == yp_test).astype(int).sum()/preds_p.shape[0])
    
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
            
            optimizer.apply_gradients(zip(grads, model_raw.trainable_weights))

            tr_loss = mse_loss(y_train_reshape, model_raw(X_train)).numpy()
            test_loss = mse_loss(y_test_reshape, model_raw(X_test)).numpy()

            pred_train = np.round(model_raw(X_train).numpy())
            acc_train = (pred_train == y_train_reshape).astype(int).sum()/y_train_reshape.shape[0]

            pred_test = np.round(model_raw(X_test).numpy())
            acc_test = (pred_test == y_test_reshape).astype(int).sum()/y_test_reshape.shape[0]

            print("epoch:", epoch, "training loss:", tr_loss, "val loss:", test_loss,
                  "acc train:", acc_train, "acc test", acc_test)

if dump_local_data:
    X_train_s = []
    y_train_s = []
    X_test_s = []
    y_test_s = []
    num_data_total = 0
    if True:
        local_data = {}
        num_data_points = {}
        for part in sorted(list(set(df.Participant.values))):

            X_loc = df.loc[df.Participant == part, ['EM0', 'EM1']].values
            y_loc = df.loc[df.Participant == part, 'Y'].values

            prop = 0.8
            index = np.random.choice(range(X_loc.shape[0]), int(np.round(prop*X_loc.shape[0])), replace=False)
            rest = [c for c in range(X_loc.shape[0]) if not (c in index)]

            X_train_loc = X_loc[index,:]
            X_test_loc = X_loc[rest,:]
            y_train_loc = y_loc[index].reshape(len(index))
            y_test_loc = y_loc[rest].reshape(len(rest))

            local_data[part] = [X_train_loc, y_train_loc, X_test_loc, y_test_loc]

            num = X_train_loc.shape[0]
            num_data_points[part] = num
            num_data_total += num

            X_train_s.append(X_train_loc)
            X_test_s.append(X_test_loc)
            y_train_s.append(y_train_loc)
            y_test_s.append(y_test_loc)

        pickle.dump(local_data, open("local_data.p", 'wb'))

    # X_train = np.concatenate(X_train_s)
    # X_test = np.concatenate(X_test_s)
    # y_train = np.concatenate(y_train_s)
    # y_test = np.concatenate(y_test_s)
    # y_train_reshape = y_train.reshape([y_train.shape[0], 1])
    # y_test_reshape = y_test.reshape([y_test.shape[0], 1])
