import os
import numpy as np
import pandas as pd
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Define base directory for images and metadata
base_skin_dir = "/home/jaden/Projects/SoftwareEng/input"

# Load the CSV file
csv_file_path = os.path.join(base_skin_dir, 'HAM10000_metadata.csv')
data = pd.read_csv(csv_file_path)

# Create a dictionary to map image IDs to their paths
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

# Mapping lesion types and adding paths to the DataFrame
lesion_type_dict = {
    'nv': 'Melanocytic nevi (nv)',
    'mel': 'Melanoma (mel)',
    'bkl': 'Benign keratosis-like lesions (bkl)',
    'bcc': 'Basal cell carcinoma (bcc)',
    'akiec': 'Actinic keratoses (akiec)',
    'vasc': 'Vascular lesions (vasc)',
    'df': 'Dermatofibroma (df)'
}

data['label'] = data['dx'].map(lambda x: {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 
                                            'akiec': 4, 'vasc': 5, 'df': 6}[x])
data['path'] = data['image_id'].map(imageid_path_dict.get)

# Handle null values (if any)
data['age'].fillna(value=int(data['age'].mean()), inplace=True)
data['age'] = data['age'].astype('int32')

# Define transformations for augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['path'].values, data['label'].values, 
                                                    test_size=0.2, random_state=1)

class SkinLesionDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # Load the image
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Create datasets
train_dataset = SkinLesionDataset(X_train, y_train, transform=transform)
test_dataset = SkinLesionDataset(X_test, y_test, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)  # Move model to the specified device (CPU or GPU)
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0  # To track correct predictions
        total_samples = 0  # To track total samples
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)  # Get class with highest score
            correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
            total_samples += labels.size(0)  # Update total samples
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        
        # Calculate training accuracy for the epoch
        train_accuracy = 100 * correct_predictions / total_samples
        
        # Print epoch metrics
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

def test_model(model, directory_path, transform):

    # Map the predicted class index to label
    label_mapping = {
        0: 'Melanocytic nevi (nv)',
        1: 'Melanoma (mel)',
        2: 'Benign keratosis-like lesions (bkl)',
        3: 'Basal cell carcinoma (bcc)',
        4: 'Actinic keratoses (akiec)',
        5: 'Vascular lesions (vasc)',
        6: 'Dermatofibroma (df)'
    }
    results = []
    
    # Iterate over all files in the directory
    for file_name in os.listdir(directory_path):
        # Construct full file path
        file_path = os.path.join(directory_path, file_name)
        
        # Check if it is an image file (you can add more extensions if needed)
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Load and preprocess the image
            img = Image.open(file_path).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)  # Add batch dimension (1, 3, 28, 28)
                
                # Inference
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                
                # Get the predicted class index
            predicted_index = predicted.item()
                
                # Get human-readable label
            if predicted_index in label_mapping:
                predicted_label = label_mapping[predicted_index]
                result_text = f"{file_name}: This is {predicted_label.lower()}."
                
                # Append result to the list
                results.append(result_text)
                print(result_text)
            
    # Return all results
    return results
    predicted_index = predicted.item()
    if predicted_index in label_mapping:
        predicted_label = label_mapping[predicted_index]
        return f"This is {predicted_label.lower()}."
    else:
        return f"Unexpected prediction index: {predicted_index}. Model might be misconfigured."


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # (3, 28, 28) -> (16, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)  # (16, 28, 28) -> (16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (32, 14, 14)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64, 7, 7)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (128, 4, 4)
        
        # Change pooling for the last conv layer
        self.pool_last = nn.MaxPool2d(2, 1)  # (128, 4, 4) -> (128, 2, 2) without shrinking too much
        
        # Fully connected layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 7)  # Adjusted to match new output size

    def forward(self, x):
        x = self.pool(self.conv1(x))
        
        x = self.pool(self.conv2(x))
        
        x = self.pool(self.conv3(x))
        
        x = self.pool_last(self.conv4(x))
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        return x
        

# Initialize model, loss function, and optimizer
model = TestModel() 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Save the model after training
model_save_path = "skin_lesion_model.pth"  # Specify the file path
torch.save(model.state_dict(), model_save_path)
print("Model saved successfully!")

model = TestModel() 

# Load the trained weights
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")

# Now you can test your model
image_path = "/home/jaden/Projects/SoftwareEng/input/multi"
result = test_model(model, image_path, transform)
print(result)