# Machine Learning Model Trainer and Predictor

This project provides a GUI-based application for training, testing, and using machine learning models with user-uploaded CSV datasets. It features preprocessing tools, visualization, and the ability to save and upload models.

---

## Features

- **GUI Interface**: Built with `tkinter` for ease of use.
- **CSV Uploading**: Upload datasets for training and prediction.
- **Data Preprocessing**: Tools to rescale, normalize, standardize, and binarize data.
- **Model Training**: Train models using various configurations.
- **Model Persistence**: Save and upload trained models with `pickle`.
- **Prediction**: Make predictions using uploaded or trained models.
- **Logging**: View logs for actions performed within the app.

---

## Installation

### Prerequisites
- Python 3.x
- Required libraries listed in `requirements.txt`

### Clone the Repository
```bash
git clone https://github.com/your-repository-link.git
cd your-project-directory
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

1. **Run the Application**:
   ```bash
   python Main.py
   ```

2. **Features Overview**:
   - **Upload CSV**: Click the "Upload .csv file" button to load a dataset.
   - **Preprocess Data**: Use buttons to rescale, normalize, standardize, or binarize data.
   - **Train Model**: Configure parameters like test size, random state, and training rounds, then train the model.
   - **Save Model**: Save the best-trained model as a `.pickle` file.
   - **Upload Model**: Upload an existing model for predictions.
   - **Make Predictions**: Input feature values and predict target outcomes using the model.

3. **Reset All**: Reset the application to its initial state.

---

## Project Structure

- **CSV_files/**: Stores temporary and preprocessed CSV files.
- **SampleData/**: Contains example datasets for testing.
- **pickle_models/**: Contains saved machine learning models.
- **Controller.py**: Handles backend logic for preprocessing, training, and prediction.
- **Main.py**: Main entry point to launch the application.
- **View.py**: GUI implementation using `tkinter`.
- **README.md**: Documentation for the project.

---

## Requirements

### Python Libraries
- tkinter
- pandas
- numpy
- scikit-learn
- matplotlib

---

## Example Workflow

1. **Upload a CSV File**:
   - Click "Upload .csv file" and select a dataset.
   - View the dataset in the "CSV Data" section.

2. **Preprocess Data**:
   - Use the buttons (e.g., "Rescale Data", "Normalize Data") to preprocess the dataset.

3. **Train a Model**:
   - Configure parameters (e.g., test size, random state).
   - Click "Test/Train Split" to train a model.
   - View accuracy and log details.

4. **Save or Upload a Model**:
   - Save the trained model using "Save Model".
   - Upload an existing model for predictions.

5. **Make Predictions**:
   - Enter input values and click "Run Model" to predict outcomes.

---

## Contribution

Contributions are welcome! Feel free to fork the repository, submit issues, or make pull requests for new features or bug fixes.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

