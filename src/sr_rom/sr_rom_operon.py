from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
from sr_rom.data.data import process_data, split_data
import matplotlib.pyplot as plt
import numpy as np
import time

# load and process data
Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")
train_data, val_data, train_val_data, test_data = split_data(
    Re, 1000*A, 1000*B, tau, a_FOM)

X_train = train_data.X.reshape(-1, 1)
y_train = train_data.y["A"][:, 0, 0]
X_val = val_data.X.reshape(-1, 1)
y_val = val_data.y["A"][:, 0, 0]
X_train_val = train_val_data.X.reshape(-1, 1)
y_train_val = train_val_data.y["A"][:, 0, 0]
X_test = test_data.X.reshape(-1, 1)
y_test = test_data.y["A"][:, 0, 0]

# training procedure for A
i = 0
j = 0

val_score = -np.inf
curr_model = None

reps = 1

# symbols = "add,mul,sub,div,fmin,fmax,aq,pow,abs,acos,asin,atan,cbrt,ceil,cos,cosh,exp,floor,log,logabs,log1p,sin,sinh,sqrt,sqrtabs,tan,tanh,square,constant,variable"
symbols = 'add,sub,mul,sin,cos,exp,log,constant,variable'

tic = time.time()
for rep in range(reps):
    reg = SymbolicRegressor(
        allowed_symbols=symbols,
        offspring_generator='basic',
        optimizer_iterations=10,
        max_length=500,
        initialization_method='btc',
        n_threads=16,
        objectives=['mse'],
        epsilon=0,
        random_state=None,
        reinserter='keep-best',
        max_evaluations=int(1e6),
        symbolic_mode=False,
        tournament_size=3,

    )

    reg.fit(X_train_val, y_train_val)
    curr_val_score = reg.score(X_test, y_test)
    if curr_val_score >= val_score:
        val_score = curr_val_score
        curr_model = reg
        print(rep, val_score)
toc = time.time()
print(toc-tic)
# print(curr_model.pareto_front_)
print(curr_model.stats_)

test_score = curr_model.score(X_test, y_test)
print(f"Test score: {test_score}")

print(f"Model: {curr_model.get_model_string(curr_model.model_)}")

k_data = np.concatenate((X_train_val, X_test)).reshape(-1, 1)
prediction = curr_model.predict(k_data)

plt.scatter(X_train_val, y_train_val,
            c="#b2df8a", marker=".", label="Training data")
plt.scatter(X_test, y_test,
            c="#b2df8a", marker="*", label="Test data")
plt.scatter(k_data, prediction, c="#1f78b4", marker='x',
            label="Best solution", linewidths=0.5)
plt.xlabel(r"$Re$")
plt.ylabel(r"$A_{ij}$")
plt.legend(loc="lower right")
plt.show()

k_sample = np.linspace(X_train[0], X_test[-1], 1001).reshape(-1, 1)
prediction = reg.predict(k_sample)

plt.scatter(X_train, y_train,
            c="#b2df8a", marker=".", label="Training data")
plt.scatter(X_test, y_test,
            c="#b2df8a", marker="*", label="Test data")
plt.plot(k_sample, prediction, c="#1f78b4", label="Best solution", linewidth=0.5)
plt.xlabel(r"$Re$")
plt.ylabel(r"$A_{ij}$")
plt.legend(loc="lower right")
plt.show()
