import os
import torch
from PIL import Image
def test_model(model, directory_path, transform):
    lesion_num_dict = {
        0: 'Melanocytic nevi (nv)',
        1: 'Melanoma (mel)',
        2: 'Benign keratosis-like lesions (bkl)',
        3: 'Basal cell carcinoma (bcc)',
        4: 'Actinic keratoses (akiec)',
        5: 'Vascular lesions (vasc)',
        6: 'Dermatofibroma (df)'
    }
    results = []
    
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(file_path).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)
            
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                predicted_index = predicted.item()
                
                if predicted_index in lesion_num_dict:
                    predicted_label = lesion_num_dict[predicted_index]
                    result_text = f"{file_name}: This is {predicted_label.lower()}."
                    results.append(result_text)
                    print(result_text)
    
