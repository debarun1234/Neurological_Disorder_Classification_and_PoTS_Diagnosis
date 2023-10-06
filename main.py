import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

# Importing data 
def import_data(file_path):
        data = []
        for filename in os.listdir(file_path):
                if filename.endswith(".csv"):
                        with open(os.path.join(file_path, filename)) as file:
                                df = pd.read_csv(file)
                                data.append(df)
        return pd.concat(data)
def preprocess_data(data):
    # Applying a high-pass filter to remove baseline wander
    b_high, a_high = signal.butter(4, 0.5, 'high', analog=False)
    if len(data) <= 8001:
        return data
    else:
        data[:, 1] = np.pad(data[:, 1], (0, len(b_high) + len(a_high)-2), mode='constant')
        data[:, 1] = signal.filtfilt(b_high, a_high, data[:, 1], axis=0)[len(b_high)-1:len(data[:, 1])-len(a_high)+1]
        # Applying a low-pass filter to remove high-frequency noise
        b_low, a_low = signal.butter(4, 35, 'low', analog=False)
        data[:, 1] = signal.filtfilt(b_low, a_low, data[:, 1], axis=0)
        return data
def split_data(data, labels):
    # Split the data and labels into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Split the training set further into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Reshape the data for the neural network
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1])

    return X_train, X_test, X_val, y_train, y_test, y_val
# Model for Classification of neurological disorder
def classify_neurological_disorder(X_train, y_train, X_test, y_test):
	# Scaling the data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	# Fitting the logistic regression model
	logistic_regression = LogisticRegression()
	logistic_regression.fit(X_train, y_train)
	# Predicting the classes
	y_pred = logistic_regression.predict(X_test)
	# Confusion matrix and classification report
	cm = confusion_matrix(y_test, y_pred)
	print("Confusion Matrix: \n", cm)
	print("Classification Report: \n", classification_report(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
# Model for Classification of PoTS using Neural Networks

def classify_POTS_nn(X_train, y_train, X_test, y_test, X_val, y_val):
    # Reshape the data to fit the model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    # Build the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=0)

    # Predict on the test data
    y_pred_nn = model.predict(X_test)
    y_test_nn = y_test

    # Get fpr and tpr for ROC curve
    fpr, tpr, _ = roc_curve(y_test_nn, y_pred_nn)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Return the true and predicted labels
    return y_test_nn, y_pred_nn
def POTS_diagnosis_statement(y_test, y_pred):
    # Compute ROC AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Determine POTS diagnosis based on ROC AUC
    if roc_auc >= 0.7:
        diagnosis = "The patient is diagnosed with POTS."
    else:
        diagnosis = "The patient is not diagnosed with POTS."

    return diagnosis, roc_auc
# Main function
def main():
    file_path = 'D:\\Desktop\\'
    data = import_data(file_path)
    data = preprocess_data(data)
    labels = np.array([0 if i < len(data) / 2 else 1 for i in range(len(data))])
    # Convert to NumPy array
    data_np = data.to_numpy()
    # Split data
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(data_np, labels)
    print("Classifying neurological disorder:")
    classify_neurological_disorder(X_train, y_train, X_test, y_test)
    print("Classifying PoTS with Random Forest Classifier:")
    classify_POTS(X_train, y_train, X_test, y_test, X_val, y_val)
    print("Classifying PoTS with Neural Network:")
    y_test_nn, y_pred_nn = classify_POTS_nn(X_train, y_train, X_test, y_test, X_val, y_val)
    # Print diagnosis statement
    diagnosis, roc_auc = POTS_diagnosis_statement(y_test_nn, y_pred_nn)
    print(diagnosis)
    print("ROC AUC: {:.3f}".format(roc_auc))
if __name__ == '__main__':
    main()
