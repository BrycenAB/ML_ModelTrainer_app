import pandas as pd
import tkinter as tk
from tkinter import filedialog
import View
import pickle
import csv
import os
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

csv_uploaded = False
os_pickle_loaded = False
formatted = False
output_type = ''
mse = 0.0


def upload_csv():
    try:
        # Save the dataframe to a global variable
        file_path = filedialog.askopenfilename()
        df = pd.read_csv(file_path)
        headers = list(df.columns)
        # if statement to check if data has been loaded before and if true deletes existing headers
        if View.datapoint_dropdown.size() > 0:
            View.datapoint_dropdown.delete(0, tk.END)
        # Create a dropdown to select the primary datapoint
        for header in headers:
            View.datapoint_dropdown.insert(tk.END, header)
        # write over current temp_csv file with new data from the df
        replace_csv('CSV_files/temp_csv.csv', file_path)
        # display_csv_data()
        global csv_uploaded
        csv_uploaded = True
        # save csv file name to View
        display_csv_name(os.path.basename(file_path))
        check_for_nan()
        display_csv_data()
        log_text(f'csv file "{os.path.basename(file_path)}" uploaded successfully\n')
    except FileNotFoundError:
        log_text('No file selected\n')
        pass


def replace_csv(existing_file, uploaded_file):
    log_text('temp_csv file data replaced with uploaded csv\n')
    # Open the uploaded file and read its contents
    with open(uploaded_file, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    # Open the existing file and write the new contents
    with open(existing_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def upload_pickle_model():
    try:
        # load model from file
        file_path = filedialog.askopenfilename()
        model = pickle.load(open(file_path, 'rb'))
        # save model to file
        pickle.dump(model, open("pickle_models/curr_best_model.pickle", "wb"))
        global os_pickle_loaded
        os_pickle_loaded = True
        display_pickle_model(file_path)
        log_text(f'Model "{os.path.basename(file_path)}" uploaded\n')
    except FileNotFoundError:
        log_text('No pickle model selected\n')
        pass


def make_prediction(model_file, input_data, target_name):   # <------ Not complete
    # Load the model from the .pickle file
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Convert the input data to a list of floats
    input_list = [float(x) for x in input_data.split(',')]
    # Perform the prediction
    prediction = model.predict([input_list])
    # Extract the target value from the prediction
    target_value = prediction

    print(target_value)


# display selected csv data in output box
def display_csv_data():
    # load dataset
    df = pd.read_csv('CSV_files/temp_csv.csv')
    # remove columns with non-numeric data
    df = df._get_numeric_data()
    # convert df to csv file
    df.to_csv('CSV_files/temp_csv.csv', index=False)
    # load csv file to View
    View.output_text.delete('1.0', tk.END)
    with open('CSV_files/temp_csv.csv', 'r') as f:
        View.output_text.insert(tk.END, f.read())


def check_predictVal_type(pred_val):
    df = pd.read_csv('CSV_files/temp_csv.csv')
    column = df[pred_val]
    # check if column is string or int/float
    if column.dtype == 'object':
        print('string')
    elif column.any() == 1 or column.any() == 0:
        return 'binary'
    else:
        # check if column is int or float
        if column.dtype == 'int64':
            return 'int'
        else:
            return 'float'


def set_return_type(type_):
    global output_type
    if type_ == 'int':
        pass


def display_pickle_model(file_name):
    if os_pickle_loaded:
        # convert file_name to just the file name
        file_name = os.path.basename(file_name)
        # replace text for View.pickle_file_name
        View.pickle_file_name.config(state='normal')
        View.pickle_file_name.delete('1.0', tk.END)
        View.pickle_file_name.insert(tk.END, f'{file_name}')
        View.pickle_file_name.config(state='disabled')
    if not os_pickle_loaded:
        View.pickle_file_name.config(state='normal')
        View.pickle_file_name.delete('1.0', tk.END)
        View.pickle_file_name.insert(tk.END, f'{file_name}')
        View.pickle_file_name.config(state='disabled')


def train_and_test_model(column_name, test_size, random_state, test_rounds):
    # load dataset
    global mse
    if not csv_uploaded:
        log_text('No csv file uploaded\n')
    if csv_uploaded:
        if column_name == 'None':
            log_text('No datapoint selected\n')
        else:
            df_to_use = pd.read_csv('CSV_files/temp_csv.csv')

            # Split the data into features (X) and target (y)
            X = df_to_use.drop(columns=[column_name])  # < ----replace with function
            y = df_to_use[column_name]
            best_score = 0

            # Split the data into training and test sets
            for _ in range(test_rounds):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Create and fit model
                model = svm.SVC(kernel='linear', C=2)
                model.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = model.predict(X_test)

                # Calculate the accuracy of the model
                score = accuracy_score(y_test, y_pred)
                acc = score

                # Calculate the mean squared error of the model
                mse = mean_squared_error(y_test, y_pred)
                if acc > best_score:
                    best_score = acc
                    # clear and Save the model to a file
                    with open("pickle_models/curr_best_model.pickle", "wb") as f:
                        pickle.dump(model, f)
            log_text(f'Model Trained successfully \n'
                     f'Predicting "{column_name}" \n'
                     f'acc of {best_score}\n'
                     f'mse of {mse}\n')
            show_best_model(best_score)
            display_pickle_model('pickle_models/curr_best_model.pickle')


# function to display the name of the csvfile uploaded in csv_file_text
def display_csv_name(name):
    # replace text for View.csv_file_name
    View.csv_file_text.config(state='normal')
    View.csv_file_text.delete('1.0', tk.END)
    View.csv_file_text.insert(tk.END, f'{name}')
    View.csv_file_text.config(state='disabled')


# function to save model from train_and_test_model to a file
def save_model(name):
    # save model to file
    model = pickle.load(open("pickle_models/curr_best_model.pickle", "rb"))
    filename = F'{name}.pickle'
    pickle.dump(model, open(filename, 'wb'))
    log_text(f'Model saved as "{name}.pickle"\n')


def show_best_model(best_score):
    # delete contents of View.accuracy_text
    View.accuracy_text.delete('1.0', tk.END)
    View.accuracy_text.insert(tk.END, f'{best_score * 100:.2f}%')


def swap_csv(return_):
    if return_:
        # load dataset
        df = pd.read_csv('CSV_files/temp_csv.csv')
        # overwrite preformatted_data.csv with df
        df.to_csv('CSV_files/preformatted_data.csv', index=False)
    else:
        # load dataset
        df = pd.read_csv('CSV_files/preformatted_data.csv')
        # overwrite temp_csv.csv with df
        df.to_csv('CSV_files/temp_csv.csv', index=False)
        return df


# rescale data between 0 and 1
def rescale_data():
    if not formatted:
        swap_csv(True)
        df = pd.read_csv('CSV_files/temp_csv.csv')
    else:
        df = swap_csv(False)
    array = df.values
    headers = df.columns
    # remove headers from array
    array = array[1:]
    # separate array into input and output components
    features_len = len(headers) - 1
    X = array[:, 0:features_len]
    Y = array[:, features_len]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(X)
    # CONVERT INPUTS and OUTPUTS to DF
    X_df = pd.DataFrame(rescaledX)
    Y_df = pd.DataFrame(Y)
    # CONCATENATE INPUTS and OUTPUTS
    df = pd.concat([X_df, Y_df], axis=1)
    # add headers back to df
    df.columns = headers
    # summarize transformed data
    np.set_printoptions(precision=3)  # < ------ set floating point precision
    # save dataframe to csv
    df.to_csv('CSV_files/temp_csv.csv', index=False)
    display_csv_data()
    log_text('Data rescaled\n')


def standardize_data():
    df = pd.read_csv('CSV_files/temp_csv.csv')
    array = df.values
    headers = df.columns
    # remove headers from array
    array = array[1:]
    # separate array into input and output components
    features_len = len(headers) - 1
    X = array[:, 0:features_len]
    Y = array[:, features_len]
    scaler = StandardScaler().fit(X)
    rescaledX = scaler.transform(X)
    # CONVERT INPUTS and OUTPUTS to DF
    X_df = pd.DataFrame(rescaledX)
    Y_df = pd.DataFrame(Y)
    # CONCATENATE INPUTS and OUTPUTS
    df = pd.concat([X_df, Y_df], axis=1)
    # add headers back to df
    df.columns = headers
    # summarize transformed data
    np.set_printoptions(precision=3)  
    # save dataframe to csv
    df.to_csv('CSV_files/temp_csv.csv', index=False)
    display_csv_data()
    log_text('Data standardized\n')


def normalize_data():
    df = pd.read_csv('CSV_files/temp_csv.csv')
    array = df.values
    headers = df.columns
    # remove headers from array
    array = array[1:]
    # separate array into input and output components
    features_len = len(headers) - 1
    X = array[:, 0:features_len]
    Y = array[:, features_len]
    scaler = Normalizer().fit(X)
    normalizedX = scaler.transform(X)
    # CONVERT INPUTS and OUTPUTS to DF
    X_df = pd.DataFrame(normalizedX)
    Y_df = pd.DataFrame(Y)
    # CONCATENATE INPUTS and OUTPUTS
    df = pd.concat([X_df, Y_df], axis=1)
    # add headers back to df
    df.columns = headers
    # summarize transformed data
    np.set_printoptions(precision=3)  
    # save dataframe to csv
    df.to_csv('CSV_files/temp_csv.csv', index=False)
    display_csv_data()
    log_text('Data normalized\n')


def binarize_data():
    df = pd.read_csv('CSV_files/temp_csv.csv')
    array = df.values
    headers = df.columns
    # remove headers from array
    array = array[1:]
    # separate array into input and output components
    features_len = len(headers) - 1
    X = array[:, 0:features_len]
    Y = array[:, features_len]
    scaler = Binarizer().fit(X)
    binarizedX = scaler.transform(X)
    # CONVERT INPUTS and OUTPUTS to DF
    X_df = pd.DataFrame(binarizedX)
    Y_df = pd.DataFrame(Y)
    # CONCATENATE INPUTS and OUTPUTS
    df = pd.concat([X_df, Y_df], axis=1)
    # add headers back to df
    df.columns = headers
    # summarize transformed data
    np.set_printoptions(precision=3)  # < ------ set floating point precision
    # save dataframe to csv
    df.to_csv('CSV_files/temp_csv.csv', index=False)
    display_csv_data()
    log_text('Data binarized\n')


# function to check a csv file for NaN values and replace them
def check_for_nan():
    df = pd.read_csv('CSV_files/temp_csv.csv')
    # check for NaN values
    if df.isnull().values.any():

        # replace NaN values with mean for column. <----- FIX ME (replace with mean not 0.0)
        df = df.fillna(0.0)
        df.to_csv('CSV_files/temp_csv.csv', index=False)
        display_csv_data()
        log_text('NaN values replaced with 0.0\n')
    else:
        # create popup window to confirm model saved
        log_text('No NaN values found\n')


def log_text(text):
    View.log_text.insert(tk.END, text)


def reset_all():
    # delete contents of View.log_text
    View.log_text.delete('1.0', tk.END)
    # delete contents of View.accuracy_text
    View.accuracy_text.delete('1.0', tk.END)
    # delete contents of View.model_text
    View.pickle_file_name.config(state='normal')
    View.pickle_file_name.delete('1.0', tk.END)
    View.pickle_file_name.insert(tk.END, "No file selected")
    View.pickle_file_name.config(state='disabled')
    # delete contents on curr_best_model.pickle
    with open('pickle_models/curr_best_model.pickle', 'wb') as f:
        pickle.dump('', f)
    # replace all data in temp_csv.csv with ''
    df = pd.DataFrame()
    df.to_csv('CSV_files/temp_csv.csv', index=False)
    # delete contents of View.csv_text
    View.output_text.delete('1.0', tk.END)
    # replace accuracy_text with 0.0
    View.accuracy_text.insert(tk.END, '0.0')
    # replace all data in preformatted_data.csv with ''
    df = pd.DataFrame()
    df.to_csv('CSV_files/preformatted_data.csv', index=False)
    # set selected_datapoint back to ''
    display_csv_name('No file selected')
    View.datapoint_dropdown.delete(0, 'end')
    global csv_uploaded
    global os_pickle_loaded
    global output_type
    csv_uploaded = False
    os_pickle_loaded = False
    output_type = ''
