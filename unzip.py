import zipfile
import os

files_to_unzip = [
    'train_data.csv.zip', 
    'train_labels.csv.zip', 
    'test_data.csv.zip', 
    'sample_submission.csv.zip'
]

for file in files_to_unzip:
    if os.path.exists(file):
        print(f"Unzipping {file}... (This might take a while for the big ones)")
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall()
        print(f"✅ Successfully extracted {file}")
    else:
        print(f"⚠️ Could not find {file}")

print("\nAll extractions complete!")