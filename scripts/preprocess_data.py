import pandas as pd
import os

# Load metadata
df = pd.read_csv(r'D:\skin_cancer\data\raw\hamdataset\HAM10000_metadata.csv')

# Rename 'dx' to 'diagnosis' for consistency
df.rename(columns={'dx': 'diagnosis'}, inplace=True)

# Define base paths for image folders
base_path_1 = r'D:\skin_cancer\data\raw\hamdataset\HAM10000_images_part_1'
base_path_2 = r'D:\skin_cancer\data\raw\hamdataset\HAM10000_images_part_2'

# Function to get the full image path
def get_image_path(image_id):
    path1 = os.path.join(base_path_1, f"{image_id}.jpg")
    path2 = os.path.join(base_path_2, f"{image_id}.jpg")
    return path1 if os.path.exists(path1) else path2

# Create the image_path column
df['image_path'] = df['image_id'].apply(get_image_path)

# Ensure the target directory exists
os.makedirs(r'D:\skin_cancer\data\processed', exist_ok=True)

# Save the processed DataFrame
df.to_csv(r'D:\skin_cancer\data\processed\data_with_paths.csv', index=False)
