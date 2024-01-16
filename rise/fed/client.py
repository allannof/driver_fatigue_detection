import os
import flwr as fl
import tensorflow as tf
import pickle
import sys

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

args = sys.argv
if len(args) == 1:
    p = 5.0
else:
    p = float(args[1])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile("adam", "mean_squared_error", metrics=["accuracy"])
model.build([1,2])

local_data = pickle.load(open("local_data.p", 'rb'))

X_train, y_train, X_test, y_test = local_data[p] 

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=1) #, batch_size=len(X_train))
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test) #, batch_size=len(X_test))
        return loss, len(X_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())
