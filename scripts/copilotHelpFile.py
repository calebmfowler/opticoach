# A Recurrent Neural Network (RNN) is a type of neural network designed for sequential data, 
# such as time series, text, or speech. Unlike traditional feedforward neural networks, RNNs 
# have connections that allow information to persist across time steps, making them well-suited 
# for tasks where context or memory is important.

# Key Features of RNNs:
#       Sequential Processing: RNNs process input sequences one step at a time, maintaining a 
#       hidden state that captures information about previous steps.
#       Shared Weights: The same weights are applied at each time step, reducing the number of 
#       parameters.
#       Applications: Commonly used for tasks like language modeling, machine translation, speech 
#       recognition, and time-series prediction.

# Use LSTMs when your data has long-term dependencies (e.g., long sequences where earlier inputs 
# influence later outputs).

import optuna # to install, enter this in the terminal: conda install -c conda-forge optuna
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, BatchNormalization
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam

# Define the input shape (e.g., sequences of length 10 with 1 feature per time step)
input_shape = (10, 1)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters to tune
    # 'units': Number of LSTM units (neurons) in the first LSTM layer
    units = trial.suggest_int('units', 32, 128)  # Range: 32 to 128
    # 'learning_rate': Learning rate for the Adam optimizer, sampled from a log-uniform distribution
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)  # Range: 0.0001 to 0.01
    # 'dropout': Fraction of input units to drop for regularization
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)  # Range: 0.1 to 0.5
    # 'batch_size': Number of samples per gradient update, chosen from a categorical list
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Options: 16, 32, 64

    # Define the model
    model = Sequential([
        # First LSTM layer with dropout and recurrent dropout for regularization
        LSTM(units, activation='tanh', return_sequences=True, input_shape=input_shape, dropout=dropout, recurrent_dropout=0.2),
        # Batch normalization to stabilize and accelerate training
        BatchNormalization(),
        # Second LSTM layer with half the units of the first layer
        LSTM(units // 2, activation='tanh', dropout=dropout, recurrent_dropout=0.2),
        # Batch normalization after the second LSTM layer
        BatchNormalization(),
        # Dense layer for the final output (e.g., regression or single-value prediction)
        Dense(1)
    ])

    # Compile the model
    # Adam optimizer with gradient clipping to prevent exploding gradients
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # Loss: Mean Squared Error, Metric: Mean Absolute Error

    # Early stopping to prevent overfitting
    # Stops training if validation loss does not improve for 5 consecutive epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    # 'history' stores the training and validation loss/metrics for each epoch
    history = model.fit(
        X_train, y_train,  # Training data
        validation_data=(X_val, y_val),  # Validation data
        epochs=50,  # Maximum number of epochs
        batch_size=batch_size,  # Batch size determined by Optuna
        callbacks=[early_stopping],  # Early stopping callback
        verbose=0  # Suppress output for faster tuning
    )

    # Return the minimum validation loss as the objective to minimize
    val_loss = min(history.history['val_loss'])
    return val_loss

# Create an Optuna study and optimize
# 'direction="minimize"' indicates that we want to minimize the validation loss
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Perform 50 trials to find the best hyperparameters

# Print the best hyperparameters and the corresponding validation loss
print(f"Best Hyperparameters: {study.best_params}")
print(f"Best Validation Loss: {study.best_value}")

# Train the final model with the best hyperparameters
# Extract the best hyperparameters from the Optuna study
best_params = study.best_params
final_model = Sequential([
    # First LSTM layer with the best number of units and dropout rate
    LSTM(best_params['units'], activation='tanh', return_sequences=True, input_shape=input_shape, dropout=best_params['dropout'], recurrent_dropout=0.2),
    # Batch normalization
    BatchNormalization(),
    # Second LSTM layer with half the units of the first layer
    LSTM(best_params['units'] // 2, activation='tanh', dropout=best_params['dropout'], recurrent_dropout=0.2),
    # Batch normalization
    BatchNormalization(),
    # Dense layer for the final output
    Dense(1)
])

# Compile the final model with the best learning rate
final_optimizer = Adam(learning_rate=best_params['learning_rate'], clipvalue=1.0)
final_model.compile(optimizer=final_optimizer, loss='mse', metrics=['mae'])

# Train the final model using the best batch size and early stopping
final_model.fit(
    X_train, y_train,  # Training data
    validation_data=(X_val, y_val),  # Validation data
    epochs=50,  # Maximum number of epochs
    batch_size=best_params['batch_size'],  # Best batch size determined by Optuna
    callbacks=[early_stopping]  # Early stopping callback
)


