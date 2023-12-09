import torch
import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
csv_file_path = 'spotify.csv'
data = pd.read_csv(csv_file_path)
data.drop(['Unnamed: 0', 'track_id', 'artists','album_name','track_name', 'track_genre'], axis=1, inplace=True)
data= data.infer_objects()
print(data.dtypes)

# classification task, take backbone

# Split the dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Compute mean and standard deviation for normalization
mean = train_data.iloc[:, :-1].mean().values
std = train_data.iloc[:, :-1].std().values

# Define a custom dataset class with normalization
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label

# Create a transform that includes normalization
transform = transforms.Compose([
    transforms.Normalize(mean, std)
])

# Create datasets and dataloaders for train and test sets
train_dataset = CustomDataset(train_data, transform=transform)
test_dataset = CustomDataset(test_data, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Now you can use train_dataloader and test_dataloader in your training/validation loop


# https://www.sciencedirect.com/science/article/abs/pii/S1568494621007584
# PCA https://towardsdatascience.com/pca-vs-autoencoders-for-a-small-dataset-in-dimensionality-reduction-67b15318dea0

# Define the autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, input_size),
            nn.Sigmoid()  # Sigmoid activation to squash the output between 0 and 1
        )

    def forward(self, x):
        # Forward pass through encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Set the input size and encoding size
input_size = 17  # Adjust according to your input data size
encoding_size = 3

# Create an instance of the autoencoder
autoencoder = Autoencoder(input_size, encoding_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}')
    for data in train_dataloader:  # Assume you have a DataLoader for your dataset
        inputs, _ = data
        inputs = inputs.view(inputs.size(0), -1)  # Flatten the input data

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = autoencoder(inputs)

        # Compute the loss
        loss = criterion(outputs, inputs)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    # Print the loss for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the encoder part for feature extraction
encoded_features = autoencoder.encoder(torch.tensor(test_dataloader))

