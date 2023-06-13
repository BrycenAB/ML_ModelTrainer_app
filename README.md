# Machine Learning Model Trainer Application
This project is a machine learning model trainer that can utilize any CSV data on your device to train and test machine learning models. It provides various features such as rescaling data, standardizing data, normalizing data, binarizing data, uploading CSV data from your computer, and saving and uploading pickle models. It has two output windows: a log window that displays common output such as errors and success messages, and an output window that displays the selected CSV file. The log window also outputs the accuracy and mean squared error (MSE) of the model.

---

# Dependencies
* Pandas
* Pickle
* Tkinter
* Numpy
* Sklearn ( pip install scikit-learn )
* Matplotlib (not currently active but code is still there)

---

# Features
* Rescaling data
* Standardizing data
* Normalizing data
* Binarizing data
* Uploading CSV data from your computer
* Saving and uploading pickle models
* Log window for displaying output messages, errors, accuracy, and MSE
* Output window for displaying the selected CSV file

---

# Usage 
1. Click the "Upload .csv File" button and select the CSV file you wish to use.
2. Select the data point to be predicted from the dropdown below the file upload button.
3. Adjust the settings according to your preferences.
4. Click the "Test/Train Split" button to train and test the model.
5. If you wish to save the model, enter the desired name into the "Pickle File Name" box and click the "Save Model" button.
6. The accuracy and MSE of the model will be displayed in the "Log" box while the CSV data will be displayed to the left in the larger box.

---
# Future Plans

* Implement Data Visualization using Matplotlib.
* Implement Data Visualization using Seaborn.
* Optimization.
