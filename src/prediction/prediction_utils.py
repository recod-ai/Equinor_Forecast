import tensorflow as tf
import numpy as np

def make_predictions(X_test, regressor):
    # Make predictions on the test set

    if isinstance(regressor, tf.keras.Model):
        y_pred = regressor.predict(X_test, verbose=0)
    else:
        y_pred = regressor.predict(X_test)


    # print('y_pred: ', y_pred)
    
    return y_pred

def inverse_pred (y_test, y_pred, X_test, regressor, scaler, max_train, cum_sum):
    
    if isinstance(regressor, tf.keras.Model):
        y_pred = y_pred.flatten()
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])
        X_test = scaler.inverse_transform(X_test)
    else:
        X_test = scaler.inverse_transform(X_test)

    
    if cum_sum:
        y_test = y_test * X_test[:, -1]
        y_pred = y_pred * X_test[:, -1]
        
    else:
        # Agora, para inverter a transformação
        y_test = max_train.inverse_transform(y_test.reshape(1, -1)).flatten()
        y_pred = max_train.inverse_transform(y_pred.reshape(1, -1)).flatten()
        
    
    return y_test, y_pred


def make_predictions_for_all_wells(X_tests, model):
    """
    Makes predictions for all wells using the provided model.

    Parameters:
    - X_tests (list of np.ndarray): List of testing features for each well.
    - model: The trained model used for making predictions.

    Returns:
    - predictions_list (list of np.ndarray): List of predictions for each well.
    """
    # Generate predictions for each well
    predictions_list = [make_predictions(X_test, model) for X_test in X_tests]
    return predictions_list


def inverse_transform_predictions(y_tests, predictions, X_tests, model, scalers, max_trains, cum_sum):
    """
    Applies inverse transformations to the predictions and true values using the provided scalers.

    Parameters:
    - y_tests (list of np.ndarray): List of true target values for each well.
    - predictions (list of np.ndarray): List of predicted values for each well.
    - X_tests (list of np.ndarray): List of testing features for each well.
    - model: The trained model used for making predictions.
    - scalers (list): List of scaler objects used for inverse transformation.

    Returns:
    - y_tests_inv (list of np.ndarray): List of inverse-transformed true target values.
    - predictions_inv (list of np.ndarray): List of inverse-transformed predicted values.
    """
    y_tests_inv = []
    predictions_inv = []

    for y_test, pred, X_test, scaler, max_train in zip(y_tests, predictions, X_tests, scalers, max_trains):
        # Apply inverse transformation using the scaler
        y_test_inv, pred_inv = inverse_pred(y_test, pred, X_test, model, scaler, max_train, cum_sum=cum_sum)
        y_tests_inv.append(y_test_inv)
        predictions_inv.append(pred_inv)

    return y_tests_inv, predictions_inv


def append_last_predictions(y_tests, predictions, y_test_list, y_pred_list):
    """
    Appends the last true and predicted values to the corresponding lists for each well.

    Parameters:
    - y_tests (list of np.ndarray): List of true target values for each well.
    - predictions (list of np.ndarray): List of predicted values for each well.
    - y_test_list (list of list): Accumulated list of true target values for each well.
    - y_pred_list (list of list): Accumulated list of predicted values for each well.

    Returns:
    - None (modifies y_test_list and y_pred_list in place).
    """
    for i, (y_test, pred) in enumerate(zip(y_tests, predictions)):
        # Append the last value from y_test and predictions to the lists
        y_test_list[i].append(y_test[-1])
        y_pred_list[i].append(pred[-1])