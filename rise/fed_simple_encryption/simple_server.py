import eventlet
eventlet.monkey_patch()
import json
from flask import Flask, request
from flask_socketio import SocketIO
import time
import secrets
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import sys
from distutils.util import strtobool as strtobool
from Pyfhel import Pyfhel

# collect some stats: size of model updates, elapsed time and
# elapsed time excluding waiting time
collected_stats = {'size': [], 'time_tot': [], 'time': []}
stats_labels = {'size': 'Package size (bytes)', 'time_tot': 'Elapsed time (s)',
               'time': 'Elapsed time excluding waiting (s)'}

# Boolean argument use_encryption True/False should be supplied on command line
# python simple_server.py True
# Choose same value for use_encryption in server and clients.

args = sys.argv[1:]

if len(args) < 1:
    print("Supply argument use_encryption.")
    sys.exit()

# use homomorphic encryption
use_encryption = bool(strtobool(args[0]))

if use_encryption:
    # generated as in prep.py
    # keyes used by server essentially for debugging by evaluating the global model
    dir_name = "HE_instance"
    HE = Pyfhel() # Empty creation
    HE.load_context(dir_name + "/context")
    HE.load_public_key(dir_name + "/pub.key")
    HE.load_secret_key(dir_name + "/sec.key")

# wait longer for response when using encryption
if not use_encryption:
    pause = 1
else:
    pause = 5
    
app = Flask(__name__)
socket = SocketIO(app, logger=True, engineio_logger=True, max_http_buffer_size=10**7)

# test data, two clients
local_data = pickle.load(open("local_data.p", 'rb'))

X_train5, y_train5, X_test5, y_test5 = local_data[5]
X_train7, y_train7, X_test7, y_test7 = local_data[7]

X_train = np.concatenate([X_train5, X_train7], axis=0)
X_test = np.concatenate([X_test5, X_test7], axis=0)
y_train = np.concatenate([y_train5, y_train7])
y_test = np.concatenate([y_test5, y_test7])

participants = local_data.keys()

metrics = ['accuracy']

global_epochs = 100

# global model used
model_fed = keras.Sequential()
model_fed.add(keras.layers.Dense(10, activation='relu'))
model_fed.add(keras.layers.Dense(10, activation='relu'))
model_fed.add(keras.layers.Dropout(0.5))
model_fed.add(keras.layers.Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=0.01)
model_fed.compile(loss='mse', optimizer=optimizer, metrics=metrics)
model_fed.build([1,2])

# shape of the neural network
model_shapes = []
for p_g in model_fed.trainable_variables:
    model_shapes.append(p_g.numpy().flatten().shape)

# loss function used
mse_loss = tf.keras.losses.MeanSquaredError()

y_train_reshape = y_train.reshape([y_train.shape[0], 1])
y_test_reshape = y_test.reshape([y_test.shape[0], 1])

y_train5_reshape = y_train5.reshape([y_train5.shape[0], 1])
y_test5_reshape = y_test5.reshape([y_test5.shape[0], 1])

y_train7_reshape = y_train7.reshape([y_train7.shape[0], 1])
y_test7_reshape = y_test7.reshape([y_test7.shape[0], 1])

num_data_points = {k: local_data[k][0].shape[0] for k in local_data}

num_data_total = sum([num_data_points[k] for k in num_data_points])

# global dictionary of received local weights
received_weights = {}

# switch if you want other serialization
def serialize(a):
    """Serialization method used for socket communication."""
    return pickle.dumps(a)

# switch if you want other serialization
def deserialize(a):
    """Serialization method used for socket communication."""
    return pickle.loads(a)

def client_update(epoch, global_weights):
    """Request updates from clients."""
    weights_serial = serialize(global_weights)
    socket.emit('server_request', {'epoch': epoch, 'global_weights': weights_serial})


@socket.on("client_response")
def receive_data(data):
    """Store updates received from clients."""
    epoch = data['epoch']
    client = data['client']
    weights_serial = data['weights']
    package_size = len(weights_serial)
    weights = deserialize(weights_serial)
    received_weights[epoch][client] = [weights, package_size]
    # print('Got message from client %s' % client, weights)

def listen():
    """Operation mode of server."""
    eventlet.sleep(10) # wait a bit first for client connection

    # initial global model
    if not use_encryption:
        global_weights = model_fed.trainable_variables
    else:
        global_weights = []
        for ii, p_g in enumerate(model_fed.trainable_variables):
            p_enc = HE.encrypt(p_g.numpy().flatten())
            global_weights.append(p_enc)  

    # global training
    for epoch in range(global_epochs):
        tic = time.time()
        wait_time = 0
        size_epoch = 0

        global_params = []
        global_params_enc = []
        received_weights[epoch] = {} # store local weights here
        client_update(epoch, global_weights) # ask clients for updates
        wait_time += pause
        eventlet.sleep(pause) # wait a bit for response
        current_weights = received_weights[epoch] # responses so far

        # demand response from all clients before proceeding
        # strong condition used in this prototype implementation
        # unnecessary to ask all clients all over again
        while len(current_weights.keys()) != len(participants):
            print("Waiting for clients.")
            client_update(epoch, global_weights) # ask again for updates
            wait_time += pause
            eventlet.sleep(pause) # wait for response
            current_weights = received_weights[epoch] # responses so far

        # parameter aggregation as weighted mean
        for part in participants:
            p_size = current_weights[part][1]
            size_epoch += 2*p_size # counting size, back and forth between server and client
            for ii, pair in enumerate(zip(current_weights[part][0], model_shapes)):
                p_l, sh = pair
                if not use_encryption:
                    if len(global_params) < ii+1:
                        global_params.append(0)
                    global_params[ii] += (num_data_points[part]/num_data_total)*p_l.numpy()
                else:
                    if len(global_params_enc) < ii+1:
                        zeros_enc = HE.encrypt(np.zeros(sh))
                        global_params_enc.append(zeros_enc)
                    # scaling of local parameters with multiplicative weight is done in clients
                    # means that only summation is done on encrypted data, not multiplication
                    global_params_enc[ii] += p_l

        # update global model
        for ii, p_g in enumerate(model_fed.trainable_variables):
            if not use_encryption:
                p_g.assign(global_params[ii])
            else:
                # decrypting to check global accuracy
                # for debugging, decryption key not available to central server in real scenario
                global_decrypt = HE.decryptFrac(global_params_enc[ii])
                global_decrypt = global_decrypt[:p_g.numpy().size]
                p_g.assign(global_decrypt.reshape(p_g.numpy().shape))

        # printing global accuracies for debugging
        # decryption key not available to central server in real scenario
        print()
        tr_loss = mse_loss(y_train, model_fed(X_train)).numpy()
        test_loss = mse_loss(y_test, model_fed(X_test)).numpy()
        pred_train = np.round(model_fed(X_train).numpy())
        acc_train = (pred_train == y_train_reshape).astype(int).sum()/y_train_reshape.shape[0]
        pred_test = np.round(model_fed(X_test).numpy())
        acc_test = (pred_test == y_test_reshape).astype(int).sum()/y_test_reshape.shape[0]
        print("epoch:", epoch, "global training loss:", tr_loss, "global test loss", test_loss, "global acc train:", acc_train, "global test acc", acc_test)
        print()

        # global weights sent to clients
        if not use_encryption:
            global_weights = model_fed.trainable_variables
        else:
            global_weights = global_params_enc

        toc = time.time()
        time_epoch_tot = toc - tic
        time_epoch = time_epoch_tot - wait_time
        collected_stats['size'].append(size_epoch)
        collected_stats['time_tot'].append(time_epoch_tot)
        collected_stats['time'].append(time_epoch)

    print("Some stats, mean over epochs:")
    for key in collected_stats.keys():
        print(stats_labels[key]+":", np.array(collected_stats[key]).mean())

    print()

    # training done
    message = "Shutting down. Training session done!"
    socket.emit('training_done', {'message': message}) # shut down clients
    eventlet.sleep(5)
    print(message)
    socket.stop()


eventlet.spawn(listen)

if __name__ == '__main__':
    socket.run(app, host='0.0.0.0', port=8080)
    
