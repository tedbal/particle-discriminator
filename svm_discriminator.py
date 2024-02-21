import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ====================================================================================================================
#                                                    CONSTANTS
# --------------------------------------------------------------------------------------------------------------------

INPUT_DIMENSION = 8

# ====================================================================================================================


class SVMSedimentDiscriminator:
    # the training data has 1=sand 0=mud
    # this will be reflected in predictions the model makes

    def __init__(self, 
                 C=50.0, 
                 gamma=0.01, 
                 model=None,
                 scaler=None, 
                 prediction_threshold=0.5):
        
        # set the prediction threshold for classification
        self.prediction_threshold = prediction_threshold

        if not model:
            # initialize the model if it's not from file
            self.model = SVC(C = C, 
                             gamma=gamma, 
                             kernel='rbf',
                             tol=1e-5)
            self.scaler = StandardScaler()

        else:
            # load the model if it is provided
            self.model = model
            self.scaler = scaler
    

    def train_model(self, x, labels, test_size=0.2, random_state=99,
                    batch_size=400, epochs=300, shuffle=True, debug=False, svm_fitting_params=None):
        # x - array of training data
        # labels - label on the corresponding row vector in x

        # fit the scaler on the training data and normalize the encoded training data
        self.scaler.fit(x)
        normalized_x = self.scaler.transform(x)

        # split the training data
        x_train, x_test, y_train, y_test = train_test_split(normalized_x, labels, test_size = test_size, random_state = random_state)

        # update the user
        print('fitting the support vector machine...')

        # define a parameter space to search over
        if svm_fitting_params == None:
            c_space = [0.1, 1, 10]
            gamma_space = [1e-3, 1e-2, 1e-1]
            svm_fitting_params = {'C': c_space, 'gamma': gamma_space}

        # perform a grid search
        clf = GridSearchCV(self.model, svm_fitting_params, refit=True, n_jobs=-1, scoring='accuracy')

        # fit the classifier (SVM)
        clf.fit(x_train, y_train)

        # update the user
        print('done')

        # make the model the best model
        self.model = clf.best_estimator_

        # show debugging information
        if debug:
            # print the grid search results
            print(pd.DataFrame(clf.cv_results_))

        # get the model predictions and threshold them
        raw_prediction = self.model.predict(x_test)
        y_tilde = np.where(raw_prediction >= self.prediction_threshold, 1, 0)

        # evaluate the accuracy score
        accuracy = accuracy_score(y_test, y_tilde)

        return accuracy
    

    def predict(self, x_vector):
        # normalize the prediction
        normalized_x_vector = self.scaler.transform(x_vector)

        # predict on the normalized prediction
        prediction = self.model.predict(normalized_x_vector)

        # threshold the prediction
        thresholded_prediction = np.where(prediction >= self.prediction_threshold, 1, 0)

        # return the prediction
        return thresholded_prediction


    def save_model(self, save_path):
        # write the save directory
        savedir = os.path.abspath(save_path)

        # check that the path exists, if the path doesn't, create it
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        
        # create the save path name
        savepath_model = os.path.join(savedir, 'model.sav')
        savepath_scaler = os.path.join(savedir, 'scaler.sav')

        # dump the model into the save file
        pickle.dump(self.model, open(savepath_model, 'wb'))
        pickle.dump(self.scaler, open(savepath_scaler, 'wb'))



def load_model(load_path):
    # interpret the model and encoder load paths
    loaddir = os.path.abspath(load_path)
    loadpath_model = os.path.join(loaddir, 'model.sav')
    loadpath_scaler = os.path.join(loaddir, 'scaler.sav')

    # load the model and the encoder
    loaded_model = pickle.load(open(loadpath_model, 'rb'))
    loaded_scaler = pickle.load(open(loadpath_scaler, 'rb'))

    # return a classifier with the model and encoder
    return SVMSedimentDiscriminator(model=loaded_model,
                                    scaler=loaded_scaler)


def parse_output_from_file(path, old_format = False):
    # make the path into an os path
    path = os.path.abspath(path)

    # get the file extension
    _, extension = os.path.splitext(path)

    # parse the file based on the extension
    if extension == '.csv':
        # create a data frame from the csv output
        df_from_processing = pd.read_csv(path)

    elif extension == '.xlsx' or extension == '.xls':
        # create a data frame from the excel output
        df_from_processing = pd.read_excel(path)

    else:
        raise Exception(f'extension {extension} unsupported: please use .csv or .xls(x) instead')

    # depending on the output format, read it in a different way
    if old_format:
        # get the indicies of the columns that should be kept
        subset_indicies = [1, 2, 3, 4, 5, 6, 7, 8] 

        # convert processing dataframe to numpy array, and index it
        data_frame = df_from_processing.to_numpy()[: subset_indicies]

    else:
        # convert it to numpy
        data_array = df_from_processing.to_numpy()[:, 1:] # exclude the "number" from the data

    # return the result
    return data_array

