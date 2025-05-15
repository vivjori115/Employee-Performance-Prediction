import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import funcs as f

st.title("Train Employee Performance Prediction Model")

if st.button("Start Training"):
    with st.spinner("Training model..."):

        # Read and prepare data
        X, Y = f.read_file()
        X = f.prep_input(X)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=10, stratify=Y)

        # Scale features
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)

        # Save the scaler
        pickle.dump(sc, open('sc.pkl', 'wb'))

        # Initialize and train model
        classifier_rfg = RandomForestClassifier(random_state=33, n_estimators=40)
        parameters = {
            'min_samples_split': [2, 3, 4, 5],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [1, 2, 3]
        }

        model_gridrf = GridSearchCV(estimator=classifier_rfg, param_grid=parameters,
                                    scoring='accuracy', cv=10, verbose=0)
        model_gridrf.fit(X_train_scaled, y_train)

        # Save the model
        pickle.dump(model_gridrf, open('model_gridrf.pkl', 'wb'))

        # Make predictions
        y_pred = model_gridrf.predict(X_test_scaled)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model trained successfully with accuracy: {accuracy:.2f}")

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    st.balloons()
