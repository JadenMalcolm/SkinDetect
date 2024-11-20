import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from config import CSV_FILE_PATH, BASE_SKIN_DIR
import cProfile
import pickle


def load_data():
    pickle_path = "data.pkl"
    
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
    data = pd.read_csv(CSV_FILE_PATH)

    # Map image IDs to their paths
    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(BASE_SKIN_DIR, '*', '*.jpg'))
    }

    # Paths and labels
    data['label'] = data['dx'].map({'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6})
    data['path'] = data['image_id'].map(imageid_path_dict.get)

    # Handle null values
    data['age'] = data['age'].fillna(value=int(data['age'].mean()))
    data['age'] = data['age'].astype('int32')
    with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

    return data

def split_data(data):
    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['path'].values, data['label'].values, test_size=0.2, random_state=1
    )
    return X_train, X_test, y_train, y_test
def create_label_mapping(csv_file):
    df = pd.read_csv(csv_file)
    label_mapping = {row['image_id']: row['dx_type'] for _, row in df.iterrows()}
    return label_mapping
def get_true_label(file_name, label_mapping):
    image_id = file_name.split('.')[0]  
    dx = label_mapping.get(image_id)    
    return lesion_num_dict.get(dx)  
