import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Data Pipeline: Load, preprocess, and split the dataset
def load_and_preprocess_data(train_path, test_path,labels_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    labels = pd.read_csv(labels_path)  # Load the separate depression labels file

    # Standardize column name for depression
    labels.rename(columns={'Depression': 'depression'}, inplace=True)
    
    train_data = train_data.merge(labels, on="id", how="left")
    test_data = test_data.merge(labels, on="id", how="left")
    def preprocess_data(df):
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna(df.mode().iloc[0], inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
        
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=['number']).columns.difference(['depression'])
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df, scaler, encoder
    
    train_processed, scaler, encoder = preprocess_data(train_data)
    test_processed, _, _ = preprocess_data(test_data)
    
    X_train = train_processed.drop(columns=['depression']).values
    y_train = train_processed['depression'].values
    X_test = test_processed.drop(columns=['depression']).values
    y_test = test_processed['depression'].values
    
    # Ensure target (y_train) is binary (0 or 1)
    y_train = (y_train > 0).astype(int)  # Ensures binary classification

    return X_train, y_train, X_test, y_test, scaler, encoder

X_train, y_train, X_test, y_test, scaler, encoder = load_and_preprocess_data(r"C:\Users\abims\Documents\playground-series-s4e11\train.csv", r"C:\Users\abims\Documents\playground-series-s4e11\test.csv",r"C:\Users\abims\Documents\playground-series-s4e11\sample_submission.csv")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)#Input Layer
        self.bn1 = nn.BatchNorm1d(128)#Hidden Layer
        self.relu = nn.ReLU()##Hidden Layer
        self.dropout1 = nn.Dropout(0.3)#Hidden Layer
        self.fc2 = nn.Linear(128, 64)#Hidden Layer
        self.bn2 = nn.BatchNorm1d(64)#Hidden Layer
        self.dropout2 = nn.Dropout(0.3)#Hiddenlayer
        self.fc3 = nn.Linear(64, 1)#Outerlayer
        self.sigmoid = nn.Sigmoid()#outerlayer 
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x

# Initialize and Train Model
input_size = X_train.shape[1]
model = MLPModel(input_size)

def train_model(model, X_train, y_train, epochs=50, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        # Forward pass
        #outputs = model(X_train_tensor)

        loss = criterion(outputs, y_train)
        # Compute Loss
        #loss = criterion(outputs, y_train_tensor)  # No more error!
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

train_model(model, X_train_tensor, y_train_tensor)

def predict_depression(user_input):
    user_df = pd.DataFrame([user_input])
   
    # Ensure all categorical columns exist
    missing_cols = [col for col in encoder.feature_names_in_ if col not in user_df.columns]
    for col in missing_cols:
        user_df[col] = "Unknown"  # Assign default value

    # One-Hot Encode categorical features (ignore unknown categories)
    encoded_array = encoder.transform(user_df[encoder.feature_names_in_])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

    # Ensure numerical columns exist before applying the scaler
    missing_numeric_cols = [col for col in scaler.feature_names_in_ if col not in encoded_df.columns]
    for col in missing_numeric_cols:
        encoded_df[col] = 0  # Assign default value for missing numeric features
    print("encoder",encoded_df)
    # Apply scaling to numerical features
    scaled_array = scaler.transform(encoded_df)
    scaled_df = pd.DataFrame(scaled_array, columns=encoded_df.columns)

    # Convert to PyTorch tensor
    user_tensor = torch.tensor(scaled_df.values, dtype=torch.float32)

    # Model prediction
    model.eval()
    with torch.no_grad():
        prediction = model(user_tensor).item()

    return 'Depression Likely' if prediction >= 0.5 else 'No Depression'


# Streamlit Application
st.title("Depression Prediction App")
st.write("Enter the required details to predict the likelihood of depression.")

age = st.number_input("Age", min_value=0, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, step=0.1)
exercise = st.selectbox("Exercise Frequency", ["Never", "Rarely", "Occasionally", "Regularly"])
family_history = st.selectbox("Family History of Depression", ["Yes", "No"])

user_input = {"age": age, "gender": gender, "sleep_hours": sleep_hours, "exercise": exercise, "family_history": family_history}

if st.button("Predict"):
    result = predict_depression(user_input)
    st.write(f"Prediction: {result}")
