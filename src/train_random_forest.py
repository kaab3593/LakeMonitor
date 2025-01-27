
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys 
import os

def train(input_folder, output_folder):
    """Train a random forest using features stored in FEATURES.CSV.
       To be used in conjunction with rdf_segment.py where the feature 
       extraction is implemented identically."""
    
    # Load the dataset from features.csv 
    data = pd.read_csv(os.path.join(input_folder, 'features.csv'))
    
    # Drop the 'Image Filename' column, as it's not needed for training
    data = data.drop(columns=['Image Filename'])

    # Check for NaNs and Infinities
    if data.isnull().sum().any() or (data == np.inf).sum().any() or (data == -np.inf).sum().any():
        print(f'Data with problem: {data}')
        print("Data contains NaNs or infinity values. Cleaning data...")
        # Replace NaNs and infinities with mean or remove rows as needed
        # Replace infinities
        data = data.replace([np.inf, -np.inf], np.nan)
        # Fill NaNs with mean of the column
        data = data.fillna(data.mean())  

    # Extract feature columns (excluding the 'Mode' column)
    X = data.drop(columns=['Mode'])

    # Extract the target labels (the 'Mode' column)
    y = data['Mode']

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier performance (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    # Save the trained model for later use
    joblib.dump(clf, os.path.join(output_folder, 'random_forest_model.pkl'))
    print("Model saved as 'random_forest_model.pkl'")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(f'Usage: python train_random_forest.py <input_folder> <output_folder>')
        print(f'Example: python train_random_forest.py /in/dir /out/dir')
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    train(input_folder, output_folder)






