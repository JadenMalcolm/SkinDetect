import torch
import os
import shutil
import cProfile
import warnings
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from data_preprocess import load_data, split_data
from dataset import SkinLesionDataset
from model import SkinModel
from train import train_model, save_model, load_model
from test import test_model, eval_model
from sanity import plot_label_distribution

warnings.filterwarnings("ignore", category=FutureWarning) # I will not be setting weights_only to true

data = load_data()
X_train, X_test, y_train, y_test = split_data(data)

plot_label_distribution(data, X_train, X_test)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = SkinLesionDataset(X_train, y_train, transform=transform)
test_dataset = SkinLesionDataset(X_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SkinModel()

if os.path.exists("skin_lesion_model.pth"):
    model.load_state_dict(torch.load('skin_lesion_model.pth'))
criterion = torch.nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.0001)

# train_model(model, train_loader, criterion, optimizer, num_epochs=2)

model_save_path = "skin_lesion_model.pth"

save_model(model, model_save_path)

model = load_model(SkinModel(), model_save_path)

image_path = "/home/jaden/Projects/SoftwareEng/input/test_loaded"

accuracy, loss = test_model(model, test_loader, criterion)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print(f"Test Loss: {loss:.4f}")

result = eval_model(model, image_path, transform)

print(result)