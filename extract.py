import zipfile
import os

def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to the specified directory.
    Args:
        zip_path (str): Path to the .zip file.
        extract_to (str): Directory where contents will be extracted.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")

def main():
    # Define paths for zip files and extraction directories
    clothing_zip = "D:\\Genaiproject\\basic-pattern-women-clothing-dataset.zip"
    batik_zip = "D:\\Genaiproject\\batik-nusantara-batik-indonesia-dataset.zip"

    clothing_extract_to = "D:\\Genaiproject\\basic-pattern-women-clothing-dataset"
    batik_extract_to = "D:\\Genaiproject\\batik-nusantara-batik-indonesia-dataset"

    # Extract datasets
    extract_zip(clothing_zip, clothing_extract_to)
    extract_zip(batik_zip, batik_extract_to)

if __name__ == "__main__":
    main()
