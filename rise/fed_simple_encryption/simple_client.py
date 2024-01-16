import socketio
import numpy as np
import sys
import pickle
from Pyfhel import Pyfhel
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from distutils.util import strtobool as strtobool

# Start simple_server.py first and then at least two instances of simple_client.py

# Two arguments to be supplied on command line: part [5 or 7] and use_encryption [True/False]
# python simple_client.py 5 True
# Choose same value for use_encryption in server and clients.

# set ip address of central server
central_server = "0.0.0.0:8080"

args = sys.argv[1:]

if len(args) < 2:
    print("Supply arguments participant and use_encryption.")
    sys.exit()

# the client
part = float(args[0])

# use homomorphic encryption
use_encryption = bool(strtobool(args[1]))

if use_encryption:
    # generated as in prep.py
    dir_name = "HE_instance"
    HE = Pyfhel() # Empty creation
    HE.load_context(dir_name + "/context")
    HE.load_public_key(dir_name + "/pub.key")
    HE.load_secret_key(dir_name + "/sec.key")

local_epochs = 1

metrics = ['accuracy']

mse_loss = tf.keras.losses.MeanSquaredError()

# test data
local_data = pickle.load(open("local_data.p", 'rb'))

X_train_loc = local_data[part][0]
y_train_loc = local_data[part][1]
y_train_loc_reshape = y_train_loc.reshape([y_train_loc.shape[0], 1])

X_test_loc = local_data[part][2]
y_test_loc = local_data[part][3]
y_test_loc_reshape = y_test_loc.reshape([y_test_loc.shape[0], 1])

num_data_points = {k: local_data[k][0].shape[0] for k in local_data}

num_data_total = sum([num_data_points[k] for k in num_data_points])

# Create socketio client to connect to central server
sio = socketio.Client()

@sio.event
def connect():
    """Connect to server."""
    print("Connection established with central server.")

@sio.event
def disconnect():
    """Disconnect from server."""
    print("Disconnected from central server.")

# switch if you want other serialization
def serialize(a):
    """Serialization method used for socket communication."""
    return pickle.dumps(a)

# switch if you want other serialization
def deserialize(a):
    """Serialization method used for socket communication."""
    return pickle.loads(a)

@sio.on('training_done')
def finish(data):
    """Shut down client requested by server."""
    print(data['message'])
    sio.disconnect()

def get_parameters(epoch, global_weights):
    """Update local parameters."""

    # local model used
    model_loc = keras.Sequential()
    model_loc.add(keras.layers.Dense(10, activation='relu'))
    model_loc.add(keras.layers.Dense(10, activation='relu'))
    model_loc.add(keras.layers.Dropout(0.5))
    model_loc.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer_loc = Adam(learning_rate=0.01)
    model_loc.compile(loss='mse', optimizer=optimizer_loc, metrics=metrics)
    model_loc.build([1,2])

    # update local model with received global weights
    if not use_encryption:
        for p_l, p_g in zip(model_loc.trainable_variables, global_weights):
            p_l.assign(p_g.numpy())
    else:
        for p_l, p_g in zip(model_loc.trainable_variables, global_weights):
            global_decrypt = HE.decryptFrac(p_g)
            global_decrypt = global_decrypt[:p_l.numpy().size]
            p_l.assign(global_decrypt.reshape(p_l.numpy().shape))

    # tr_loss = mse_loss(y_train_loc_reshape, model_loc(X_train_loc)).numpy()
    # test_loss = mse_loss(y_test_loc, model_loc(X_test_loc)).numpy()

    # pred_train = np.round(model_loc(X_train_loc).numpy())
    # acc_train = (pred_train == y_train_loc_reshape).astype(int).sum()/y_train_loc_reshape.shape[0]

    # pred_test = np.round(model_loc(X_test_loc).numpy())
    # acc_test = (pred_test == y_test_loc_reshape).astype(int).sum()/y_test_loc_reshape.shape[0]

    # print("epoch:", epoch, "local epoch", -1, "part:", part, "local training loss:", tr_loss, "local test loss:", test_loss, "local acc train:", acc_train, "local acc test", acc_test)

    # local training
    for l_epoch in range(local_epochs):

        with tf.GradientTape() as t:

            loss = mse_loss(y_train_loc_reshape, model_loc(X_train_loc))

            params = model_loc.trainable_variables

            grads = t.gradient(loss, params)

            # gradient descent step
            optimizer_loc.apply_gradients(zip(grads, model_loc.trainable_weights))

            # evaluate accuracy
            tr_loss = mse_loss(y_train_loc_reshape, model_loc(X_train_loc)).numpy()
            test_loss = mse_loss(y_test_loc, model_loc(X_test_loc)).numpy()

            pred_train = np.round(model_loc(X_train_loc).numpy())
            acc_train = (pred_train == y_train_loc_reshape).astype(int).sum()/y_train_loc_reshape.shape[0]

            pred_test = np.round(model_loc(X_test_loc).numpy())
            acc_test = (pred_test == y_test_loc_reshape).astype(int).sum()/y_test_loc_reshape.shape[0]

            print("epoch:", epoch, "local epoch", l_epoch, "part:", part, "local training loss:", tr_loss, "local test loss:", test_loss, "local acc train:", acc_train, "local acc test", acc_test)

    # updated local parameters
    if not use_encryption:
        output = model_loc.trainable_variables
    else:
        output = []
        for ii, p_l in enumerate(model_loc.trainable_variables):
            update_enc = HE.encrypt((num_data_points[part]/num_data_total)*p_l.numpy().flatten())
            output.append(update_enc)

    return output


def update_weights(data):
    """Model update requested by server."""

    epoch = data['epoch']
    global_weights = deserialize(data['global_weights'])
    local_weights = get_parameters(epoch, global_weights)
    local_weights_serial = serialize(local_weights)

    # message to server
    message = {'epoch': epoch, 'client': part, 'weights': local_weights_serial}
    sio.emit('client_response', message)


if __name__ == '__main__':
    # Setup socket callback
    sio.on("server_request", update_weights)

    # Connect to socket
    sio.connect("http://" + central_server)
