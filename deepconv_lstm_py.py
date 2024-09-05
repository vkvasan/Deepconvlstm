import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the datasets from the provided files
X_train = np.load('./mesl_data/x_train.npy')
X_test = np.load('./mesl_data/x_test.npy')
y_train = np.load('./mesl_data/y_train.npy')
y_test = np.load('./mesl_data/y_test.npy')

# Verify the shapes of the data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Check the unique values in y_train and y_test
print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_test:", np.unique(y_test))

# Ensure all labels are within the correct range
assert np.all((y_train >= 0) & (y_train < 26)), "y_train contains out-of-range values"
assert np.all((y_test >= 0) & (y_test < 26)), "y_test contains out-of-range values"

# Normalize data
def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

# Normalize the data
X_train = normalize(X_train)
X_test = normalize(X_test)

# Convert data to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)  # Ensuring labels are in int64
y_test = y_test.astype(np.int64)    # Ensuring labels are in int64

# Reshape data for the model
X_train = X_train.reshape((-1, 1, X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((-1, 1, X_test.shape[1], X_test.shape[2]))

print(f"Train data shape after reshape: {X_train.shape}")
print(f"Test data shape after reshape: {X_test.shape}")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Hardcoded parameters for the new dataset
NB_SENSOR_CHANNELS = 9
NUM_CLASSES = 26  # Updated number of classes
SLIDING_WINDOW_LENGTH = 150
BATCH_SIZE = 16
NUM_FILTERS = 64
FILTER_SIZE = 5
NUM_UNITS_LSTM = 128
LEARNING_RATE = 0.0001
NUM_EPOCHS = 150

# Define the network
class DeepConvLSTM(nn.Module):
    def __init__(self):
        super(DeepConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, NUM_FILTERS, (FILTER_SIZE, 1))
        self.conv2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1))
        self.conv3 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1))
        self.conv4 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1))
        self.lstm1 = nn.LSTM(NUM_FILTERS * NB_SENSOR_CHANNELS, NUM_UNITS_LSTM, batch_first=True)
        self.lstm2 = nn.LSTM(NUM_UNITS_LSTM, NUM_UNITS_LSTM, batch_first=True)
        self.fc = nn.Linear(NUM_UNITS_LSTM, NUM_CLASSES)

        # Weight initialization
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        return x
    

model = DeepConvLSTM()
    # Create datasets and dataloaders
train_data = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_data = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

state_dict = torch.load('weights/DeepConvLSTM_trained_mesl_data.pth', map_location=torch.device('cpu'))

# Load the state dictionary into the model
model.load_state_dict(state_dict)

device = torch.device('cpu')
model.to(device)
# Evaluation
model.eval()
test_pred = []
test_true = []
batch_times = []
sample_times = []

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to('cpu'), targets.to('cpu')

        # Start time for the batch
        start_time = time.time()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # End time for the batch
        end_time = time.time()

        # Record the inference time for the batch
        batch_inference_time = end_time - start_time
        batch_times.append(batch_inference_time)

        # Calculate per-sample time for the batch
        batch_size = inputs.size(0)
        per_sample_time = batch_inference_time / batch_size
        sample_times.append(per_sample_time)

        test_pred.extend(preds.cpu().numpy())
        test_true.extend(targets.cpu().numpy())

        # Print per-batch time and per-sample time for the current batch
        #print(f"Batch {batch_idx + 1}:")
        #print(f"  Batch Inference Time: {batch_inference_time:.4f} seconds")
        #print(f"  Per-Sample Inference Time: {per_sample_time:.8f} seconds")

# Calculate metrics
accuracy = accuracy_score(test_true, test_pred)
macro_precision = precision_score(test_true, test_pred, average='macro')
macro_recall = recall_score(test_true, test_pred, average='macro')
macro_f1 = f1_score(test_true, test_pred, average='macro')

# Calculate average batch time and average per-sample time
average_batch_time = sum(batch_times) / len(batch_times)
average_per_sample_time = sum(sample_times) / len(sample_times)

# Results presentation
print(f"\nOverall Test Results:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Average Batch Inference Time: {average_batch_time:.4f} seconds")
print(f"Average Per-Sample Inference Time: {average_per_sample_time:.8f} seconds")
