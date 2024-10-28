import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Path to dataset folder
dataset_path = r'C:\Users\Dilshaan\OneDrive\Pictures\Camera Roll\Desktop\archive\Actor_04'

# Function to extract MFCC features from an audio file
def extract_mfcc_features(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare dataset
X = []  # Features
y = []  # Labels

# Iterate through all files in the dataset folder
for file_name in os.listdir(dataset_path):
    if file_name.endswith('.wav'):
        print(f"Processing file: {file_name}")
        
        # Extract emotion from the filename (assuming labels are embedded in filename)
        # Example: "03-01-01-01-01-01-01.wav" -> label extracted from the 6th component
        label = int(file_name.split("-")[2])  # Adjust as per your filename pattern
        
        # Extract MFCC features
        file_path = os.path.join(dataset_path, file_name)
        mfcc_features = extract_mfcc_features(file_path)
        
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train an SVM classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")
