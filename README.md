Employee Performance Prediction App
===================================

Overview
--------
This is a Streamlit web application that predicts employee performance ratings based on various input features.  
The prediction is powered by a Random Forest model trained on employee data.

Features
--------
- User-friendly form to input employee details.
- Encodes categorical features automatically.
- Scales numeric inputs before prediction.
- Displays predicted performance rating instantly.

Installation
------------
1. Clone the repository or download the source code.
2. Make sure you have Python 3.7+ installed.
3. Install required packages using pip::

   pip install -r requirements.txt

Files
-----
- ``app.py`` : Streamlit application file to run the web app.
- ``model.py`` : Script used to train and save the model.
- ``funcs.py`` : Helper functions for data processing.
- ``data.csv`` : Sample dataset file.
- ``model_gridrf.pkl`` : Pickled trained Random Forest model.
- ``sc.pkl`` : Pickled StandardScaler used for scaling features.

Usage
-----
To run the Streamlit app, execute the following command in your terminal::

   streamlit run app.py

Open your browser and go to the URL provided by Streamlit (usually http://localhost:8501).

Input the employee details on the form and click 'Predict' to see the performance rating prediction.

Notes
-----
- Make sure the model and scaler pickle files are in the same directory as `app.py`.
- The sample data ``data.csv`` must have the necessary columns used in prediction.
- If you update the model or scaler, replace the respective `.pkl` files.

Contributing
------------
Feel free to fork and submit pull requests. For major changes, please open an issue first to discuss.

License
-------
This project is licensed under the MIT License.

Contact
-------
For questions or feedback, contact: viv.jori11@gmail.com

