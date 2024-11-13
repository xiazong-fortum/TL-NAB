import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from LSTMAutoencoder import SimpleLSTMAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the LSTM Autoencoder model based on your provided class


def load_model(model_file):
    # Create an instance of the model
    # model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    # Load the state dictionary into the model
    # model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model = torch.load(model_file).to(device)
    model.eval()
    return model

def generate_anomaly_score(inputData, model_file, scaler, sequence_length, batch_size=64):
    # Load the model
    model = torch.load(model_file, map_location=device, weights_only=False)
    model.eval()
    
    # Standardize input data as training data was standardized
    oil_pressure_columns = inputData.columns[inputData.columns.str.startswith('oil_pressure')].values
    for column in oil_pressure_columns:
        inputData.loc[:, column] = scaler.transform(inputData.loc[:, column].values.reshape(-1,1)).flatten()
    # Transform input data to sequences and create DataLoader
    data_loader = transform_data(inputData, sequence_length=sequence_length, batch_size=batch_size)

    # Initialize a list to store sum of scores and counts for averaging
    score_sums = np.zeros(inputData.shape[0])
    score_counts = np.zeros(inputData.shape[0])

    with torch.no_grad():
        for i, (input_tensor,) in enumerate(data_loader):
            input_tensor = input_tensor.to(device)
            decoded = model(input_tensor)
            reconstruction_errors = torch.mean((decoded - input_tensor) ** 2, dim=2).cpu().numpy()  # Shape (batch_size, sequence_length)
            
            # Distribute reconstruction errors into the appropriate positions in score_sums and score_counts
            for j, error_sequence in enumerate(reconstruction_errors):
                start_index = i * batch_size + j
                for k in range(sequence_length):
                    if start_index + k < len(score_sums):
                        score_sums[start_index + k] += error_sequence[k]
                        score_counts[start_index + k] += 1

    # Calculate the average anomaly scores by dividing sums by counts
    all_anomaly_scores = score_sums / (score_counts + 1e-6)  # Avoid division by zero
    
    #Normalize the anomaly scores to [0,1]
    min_score = np.min(all_anomaly_scores)
    max_score = np.max(all_anomaly_scores)
    all_anomaly_scores = (all_anomaly_scores - min_score) / (max_score - min_score + 1e-6)  # Avoid division by zero

    assert len(all_anomaly_scores) == inputData.shape[0]
    return all_anomaly_scores

def transform_data(X, sequence_length, batch_size=32):
    # Convert the input dataframe to a tensor
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    
    # Calculate the number of sequences that can be created
    num_samples = X_tensor.shape[0] - sequence_length + 1
    num_features = X_tensor.shape[1]
    
    # Create sliding window sequences with a step of 1
    X_sequences = torch.zeros((num_samples, sequence_length, num_features), dtype=torch.float32)
    for i in range(num_samples):
        X_sequences[i] = X_tensor[i:i + sequence_length]
    
    # Create a TensorDataset and DataLoader for batch processing
    dataset = TensorDataset(X_sequences)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader

def process_and_save_anomaly_scores(data_files, model_files, root_dir, scaler, sequence_length):
    for model_file in model_files:
        for file in data_files:
            # Load data from CSV with timestamp as index
            df = pd.read_csv(os.path.join(root_dir, file), index_col=['timestamp'])
            
            # Only use numeric columns for model input
            # numeric_data = df.select_dtypes(include=[np.number])
            # inputData = numeric_data.values  # Convert DataFrame to numpy array
            orginal_df = df.copy(deep=True)
            # Generate anomaly scores
            anomaly_scores = generate_anomaly_score(df, model_file, scaler, sequence_length)
            
            # Add the anomaly scores as a new column to the DataFrame
            model_name = os.path.splitext(os.path.basename(model_file))[0]
            orginal_df['anomaly_score'] = anomaly_scores
            
            # Save the DataFrame with anomaly scores as a new CSV file
            output_file = os.path.join('./results', file.replace('.csv', f'_{model_name}_scored.csv'))
            orginal_df.to_csv(output_file, index=True)  # Preserve index (timestamps) in output file
            print(f"Anomaly scores saved to {output_file}")
    return orginal_df
import json

def main():
    # Root directory
    root_dir = './transfer_learning'

    # File paths relative to root directory
    model_files = ['/model/lstm_small.pth']  # Add more model files as needed
    data_files = ['/data/inter_leakage.csv', '/data/pump_failure.csv']
    # scaler = StandardScaler()
    
    # Run the anomaly detection and save results
    process_and_save_anomaly_scores(data_files, model_files, root_dir, scaler, sequence_length)

    
if __name__ == "__main__":
    main()

