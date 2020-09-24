import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, LeakyReLU, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split
import os
import plaidml

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

data = pd.read_csv('DataTrain_week1.csv', index_col='date')

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = split_sequence(data['bidclose'].values, 10)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=0.6, test_size=0.2, shuffle=False)

def build_model(hp):
    model = Sequential()
    model.add(Conv1D(hp.Int('Conv1D_input', min_value=32, max_value=128, step=32), kernel_size=2,
                     input_shape=(10, 1)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Dropout(hp.Float('Conv1D_input_dropout', min_value=0.1, max_value=0.4, step=0.1)))

    for i in range(hp.Int('n_Conv_layers', 0, 2)):
        model.add(Conv1D(hp.Int(f'Conv1D_{i}_units', min_value=32, max_value=128, step=32), kernel_size=2,
                         input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
        model.add(Dropout(hp.Float(f'Conv1D_{i}_units_dropout', min_value=0.1, max_value=0.4, step=0.1)))

    model.add(Bidirectional(LSTM(hp.Int('LSTM', min_value=32, max_value=128, step=32))))
    model.add(Dropout(hp.Float('LSTM_dropout', min_value=0, max_value=0.6, step=0.1)))

    for i in range(hp.Int('n_Dense_layers', 0, 2)):
        model.add(Dense(units=hp.Int(f'Dense_{i}_units', min_value=32, max_value=128, step=32),
                        activation=LeakyReLU(alpha=0.1)))
        model.add(Dropout(hp.Float(f'Dense_{i}_dropout', min_value=0, max_value=0.8, step=0.05)))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
    return model


tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=1,
    num_initial_points=1,
    overwrite=True)

tuner.search(x=X_train,
             y=y_train,
             epochs=500,
             batch_size=256,
             validation_data=(X_test, y_test))

best_model = tuner.get_best_models(num_models=1)[0]

yhat=best_model.predict(X_train)
plt.plot(yhat, linewidth=0.75, label='Yhat')
plt.plot(y_test, linewidth=0.75, color='k', label='y_actual')
plt.legend()
plt.show()
