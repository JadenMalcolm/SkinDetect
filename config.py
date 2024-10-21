import os

BASE_SKIN_DIR = "/home/jaden/Projects/SoftwareEng/input/"

CSV_FILE_PATH = os.path.join(BASE_SKIN_DIR, 'HAM10000_metadata.csv')

# Lesion mapping
LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi (nv)',
    'mel': 'Melanoma (mel)',
    'bkl': 'Benign keratosis-like lesions (bkl)',
    'bcc': 'Basal cell carcinoma (bcc)',
    'akiec': 'Actinic keratoses (akiec)',
    'vasc': 'Vascular lesions (vasc)',
    'df': 'Dermatofibroma (df)'
}
