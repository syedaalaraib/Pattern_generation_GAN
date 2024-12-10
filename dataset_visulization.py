import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class TextileDataset(Dataset):
    def __init__(self, batik_dir, clothing_dir, transform=None):
        """
        Combined Dataset for Batik and Clothing Patterns
        
        Args:
            batik_dir (str): Directory with Batik images
            clothing_dir (str): Directory with Clothing Pattern images
            transform (callable, optional): Optional transform to be applied
        """
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Process Batik Dataset
        if os.path.exists(batik_dir):
            self._process_dataset(batik_dir, dataset_type='batik')
        
        # Process Clothing Pattern Dataset
        if os.path.exists(clothing_dir):
            self._process_dataset(clothing_dir, dataset_type='clothing')
        
        # Create metadata
        self.metadata = pd.DataFrame({
            'file_path': self.image_paths,
            'label': self.labels
        })
    
    def _process_dataset(self, directory, dataset_type):
        """
        Process images from a given directory.
        Recursively traverses nested directories to find image files.
        """
        for root, _, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process images
                    try:
                        with Image.open(full_path) as img:
                            self.image_paths.append(full_path)
                            
                            # Use the folder name as the label
                            label = os.path.basename(os.path.dirname(full_path))
                            self.labels.append(f"{dataset_type}_{label}")
                    except Exception as e:
                        print(f"Error processing {full_path}: {e}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Generate one sample of data
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def visualize_dataset(batik_dir, clothing_dir):
    """
    Comprehensive Data Visualization
    """
    # Create dataset without transforms for visualization
    dataset = TextileDataset(
        batik_dir, 
        clothing_dir, 
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    
    # Metadata and Label Analysis
    metadata_df = dataset.metadata

    # Ensure dataset is not empty
    if metadata_df.empty:
        print("No valid images found in the dataset directories.")
        return None

    # Class Distribution
    label_counts = metadata_df['label'].value_counts()
    if label_counts.empty:
        print("No labels found for visualization.")
        return dataset
    
    # 1. Bar Chart for Class Distribution
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    label_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Pattern Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 2. Pie Chart for Class Distribution
    plt.subplot(122)
    plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
    plt.title('Class Distribution Percentage')
    plt.tight_layout()
    
    plt.savefig('class_distribution.png')
    plt.show()
    
    # 3. Sample Images
    def plot_samples(num_samples=5):
        unique_labels = metadata_df['label'].unique()
        
        plt.figure(figsize=(20, 4))
        for i, label in enumerate(unique_labels[:num_samples]):  # Limit to num_samples
            sample_img_path = metadata_df[metadata_df['label'] == label]['file_path'].iloc[0]
            
            plt.subplot(1, num_samples, i + 1)
            img = Image.open(sample_img_path)
            plt.imshow(img)
            plt.title(label)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png')
        plt.show()

    plot_samples()
    
    # Print Dataset Statistics
    print("\nDataset Statistics:")
    print(f"Total Images: {len(dataset)}")
    print("\nClass Distribution:")
    print(label_counts)
    
    # Class Imbalance Analysis
    print("\nClass Imbalance Analysis:")
    total_images = len(dataset)
    for label, count in label_counts.items():
        percentage = (count / total_images) * 100
        print(f"{label}: {count} images ({percentage:.2f}%)")
    
    return dataset


def main():
    # Paths to the datasets
    batik_dir = "D:\\Genaiproject\\batik-nusantara-batik-indonesia-dataset"
    clothing_dir = "D:\\Genaiproject\\basic-pattern-women-clothing-dataset\\basicpattern_3800"
    
    # Visualize dataset
    dataset = visualize_dataset(batik_dir, clothing_dir)


if __name__ == "__main__":
    main()
