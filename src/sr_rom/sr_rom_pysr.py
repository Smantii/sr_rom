import numpy as np
from pysr import PySRRegressor
from sr_rom.data.data import process_data, split_data
import dctkit as dt
import time

Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")
train_data, val_data, train_val_data, test_data = split_data(
    Re, 1000*A, 1000*B, tau, a_FOM)

X_train = train_val_data.X.reshape(-1, 1)
y_train = train_val_data.y["A"][:, 0, 2]
X_test = test_data.X.reshape(-1, 1)
y_test = test_data.y["A"][:, 0, 2]

model = PySRRegressor(
    populations=1,
    population_size=1000,
    niterations=70,  # < Increase me for better results
    binary_operators=["+", "*", "-"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "log"
        # ^ Custom operator (julia syntax)
    ],
    maxsize=150,
    # ^ Custom loss function (julia syntax)
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    batching=False,
    multithreading=True,
    turbo=True
)

tic = time.time()
model.fit(X_train, y_train)
toc = time.time()
print(toc - tic)
