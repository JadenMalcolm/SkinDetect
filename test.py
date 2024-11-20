import os
from PIL import Image
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
LESION_NUM_DICT = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3,
    'akiec': 4, 'vasc': 5, 'df': 6
}
def test_model(model, test_loader, criterion):
    model.eval()  
    total_correct = 0
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad(): 
        for images, labels, paths in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss
    
def eval_model(model, directory_path, transform):
    # Invert the dictionary to get index to label for plotting
    index_to_label = {v: k for k, v in LESION_NUM_DICT.items()}
    labels = list(LESION_NUM_DICT.keys())
    
    true_labels = []
    predicted_labels = []

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            true_label_str = file_name.split('_')[0].lower()
            true_label = LESION_NUM_DICT.get(true_label_str)

            if true_label is not None:
                img = Image.open(file_path).convert('RGB')
                img = transform(img)
                img = img.unsqueeze(0)
                
                with torch.no_grad():
                    output = model(img)
                    _, predicted = torch.max(output, 1)
                    predicted_index = predicted.item()
                    
                    true_labels.append(true_label)
                    predicted_labels.append(predicted_index)

                    predicted_label = index_to_label[predicted_index]
                    result_text = f"{file_name}: This is {predicted_label.lower()}."

    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    mcc = matthews_corrcoef(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    classification_report_str = classification_report(true_labels, predicted_labels, target_names=labels)

    tn = cm.sum(axis=1) - cm.diagonal()
    fp = cm.sum(axis=0) - cm.diagonal()
    specificity = tn / (tn + fp)
    
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nClassification Report:\n", classification_report_str)
    print("\nSpecificity per class:")
    for label, spec in zip(labels, specificity):
        print(f"{label}: {spec:.4f}")

    return cm, mcc, precision, recall, f1, specificity
