import pandas as pd
import os

csv_file = 'HAM10000_metadata.csv'
data = pd.read_csv(csv_file)

image_dir = 'test_loaded'

for index, row in data.iterrows():
    image_id = f"{row[1]}.jpg" 
    lesion_type = row[2] 
    
    current_file_path = os.path.join(image_dir, image_id)
    new_file_name = f"{lesion_type}_{image_id}"
    new_file_path = os.path.join(image_dir, new_file_name)
    
    if os.path.exists(current_file_path):
        os.rename(current_file_path, new_file_path)
        print(f"Renamed {image_id} to {new_file_name}")
    else:
        print(f"File {image_id} does not exist in the directory.")

