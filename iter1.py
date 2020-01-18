import time
import warnings
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut

# Global varaiables used for bootstrapping
OBSERVATIONS_PER_BIN = 11
BOOTSTRAP_ITERATIONS = 50
PERMUTATION_ITERATIONS = 20

# General global variables
TIME_INTERVAL = 1288
NUMBER_OF_FEATURES = 306
AGG_PATH = "./agg"
RESULT_PATH = f"results/{int(time.time())}"
COLUMN_LABELS = [str(i) for i in range(306)].append("y")

#Grid Search for SVM
SVM_PARAMETERS = {
    'C': (1000000000, 100000, 1000, 100, 10, 1),
    'gamma': (1, .01, .0001, 'auto'),
}

def generate_decoded_time_series():
    accuracy_list = []
    for i in range(TIME_INTERVAL):
        file_path = AGG_PATH + "/" + str(i) + ".csv"
        df = pd.read_csv(file_path)
        normalized_df = get_normalized_df(df)
        accuracy = generate_model(normalized_df)
        accuracy_list.append(accuracy)

    ser = pd.Series(accuracy_list)
    ser.to_csv(RESULT_PATH)
    return accuracy_list

def generate_model(df):
    accuracy_list = []
    correct_num, incorrect_num = get_class_distribution(df)
    sample_size = min(correct_num, incorrect_num)

    for bootstrap_iteration in range(BOOTSTRAP_ITERATIONS):
        correct_sample, incorrect_sample = get_samples(df, sample_size)
        correct_X, correct_y = get_feature_target_split(correct_sample)
        incorrect_X, incorrect_y = get_feature_target_split(incorrect_sample)
        kf = KFold()
        for train_index, test_index in kf.split(correct_X):
            #Creating training and testing feature and target sets
            correct_train_X, correct_test_X  = correct_X[train_index], correct_X[test_index]
            incorrect_train_X, incorrect_test_X = incorrect_X[train_index], incorrect_X[test_index]
            correct_train_y, correct_test_y = correct_y[train_index], correct_y[test_index]
            incorrect_train_y, incorrect_test_y = incorrect_y[train_index], incorrect_y[test_index]
            train_X = np.concatenate((correct_train_X, incorrect_train_X))
            train_y = np.concatenate((correct_train_y, incorrect_train_y))
            train_X, train_y = shuffle(train_X, train_y)
            test_X = np.concatenate((correct_test_X, incorrect_test_X))
            test_y = np.concatenate((correct_test_y, incorrect_test_y))
            test_X, test_y = shuffle(test_X, test_y)

            #Generate the model and grid search for optimal parameter set
            svm = SVC()
            clf = GridSearchCV(svm, SVM_PARAMETERS)
            clf.fit(train_X, train_y)
            predictions = clf.predict(test_X)

            accuracy = accuracy_score(test_y, predictions)
            accuracy_list.append(accuracy)

    average_accuracy = mean(accuracy_list)

    return average_accuracy

def get_class_distribution(df):
    n = len(df.index)
    correct_num = len(df[df.y==0].index)
    incorrect_num = n - correct_num
    return correct_num, incorrect_num

def get_samples(df, sample_size):
    correct_sample = df[df.y==0].sample(sample_size)
    incorrect_sample = df[df.y==1].sample(sample_size)
    return correct_sample, incorrect_sample

def get_normalized_df(df):
    y = df["y"].values
    X = df.drop("y", axis=1).values
    normalized_X = normalize(X)
    df = pd.DataFrame(normalized_X)
    df["y"] = y
    return df

def get_feature_target_split(df):
    y = df["y"].values
    X = df.drop("y", axis=1).values
    return X, y

def shuffle(X, y):
    y = y.reshape((X.shape[0], 1))
    data = np.concatenate((X, y), axis=1)
    shuffled_data = pd.DataFrame(data).sample(frac=1).values
    X = shuffled_data[:, :-1]
    y = shuffled_data[:, -1:].reshape(len(y),)
    return X, y

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
generate_decoded_time_series()
