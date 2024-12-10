import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from PIL import Image

# -------------------------------------
# Helper Function to Load Logs
# -------------------------------------
def load_logs(file_path, label):
    """Load logs from a CSV file and add a configuration label."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    logs = pd.read_csv(file_path)
    logs['Configuration'] = label  # Add label for identification
    return logs

# -------------------------------------
# Function to Calculate FID Score
# -------------------------------------
def calculate_fid(real_images_path, generated_images, transform):
    """
    Calculate FID score between real and generated images.
    Args:
        - real_images_path (str): Path to directory containing real images
        - generated_images (Tensor): Generated images tensor
        - transform (callable): Transform function to preprocess the images
    Returns:
        - fid_score (float): Calculated FID score
    """
    fid = FrechetInceptionDistance()
    
    # Load real images
    real_images = []
    for img_name in os.listdir(real_images_path):
        img_path = os.path.join(real_images_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        real_images.append(img)
    
    # Convert list to tensor
    real_images = torch.stack(real_images).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Update FID metric with real and generated images
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    
    # Calculate and return FID score
    fid_score = fid.compute()
    return fid_score.item()

# -------------------------------------
# Load Experiment Logs
# -------------------------------------
logs_batchnorm = load_logs('logs_batchnorm_smooth.csv', 'With BatchNorm')
logs_no_batchnorm = load_logs('logs_no_batchnorm_smooth.csv', 'Without BatchNorm')
logs_lr_low = load_logs('logs_lr_0.0001_smooth.csv', 'LR: 0.0001')
logs_lr_high = load_logs('logs_lr_0.001_smooth.csv', 'LR: 0.001')
logs_momentum_low = load_logs('logs_momentum_0.5_smooth.csv', 'Momentum: 0.5')
logs_momentum_high = load_logs('logs_momentum_0.9_smooth.csv', 'Momentum: 0.9')

# Combine all logs for accuracy comparison
all_logs = [logs_batchnorm, logs_no_batchnorm, logs_lr_low, logs_lr_high, logs_momentum_low, logs_momentum_high]
all_logs = [log for log in all_logs if log is not None]  # Filter out missing files

# -------------------------------------
# Visualization: Batch Normalization
# -------------------------------------
if logs_batchnorm is not None and logs_no_batchnorm is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(logs_batchnorm['Epoch'], logs_batchnorm['Generator Loss'], label='With BatchNorm')
    plt.plot(logs_no_batchnorm['Epoch'], logs_no_batchnorm['Generator Loss'], label='Without BatchNorm')
    plt.title('Impact of Batch Normalization on Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.legend()
    plt.savefig('batchnorm_impact.png')
    plt.show()

# -------------------------------------
# Visualization: Learning Rate
# -------------------------------------
if logs_lr_low is not None and logs_lr_high is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(logs_lr_low['Epoch'], logs_lr_low['Generator Loss'], label='LR: 0.0001')
    plt.plot(logs_lr_high['Epoch'], logs_lr_high['Generator Loss'], label='LR: 0.001')
    plt.title('Impact of Learning Rate on Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.legend()
    plt.savefig('learning_rate_impact.png')
    plt.show()

# -------------------------------------
# Visualization: Momentum
# -------------------------------------
if logs_momentum_low is not None and logs_momentum_high is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(logs_momentum_low['Epoch'], logs_momentum_low['Discriminator Loss'], label='Momentum: 0.5')
    plt.plot(logs_momentum_high['Epoch'], logs_momentum_high['Discriminator Loss'], label='Momentum: 0.9')
    plt.title('Impact of Momentum on Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Loss')
    plt.legend()
    plt.savefig('momentum_impact.png')
    plt.show()

# -------------------------------------
# Analysis of Overall Accuracy
# -------------------------------------
if all_logs:
    accuracy_data = {
        'Configuration': [log['Configuration'].iloc[0] for log in all_logs],
        'Accuracy': [log['Accuracy'].iloc[-1] for log in all_logs]  # Final epoch accuracy
    }

    accuracy_df = pd.DataFrame(accuracy_data)

    plt.figure(figsize=(10, 6))
    plt.bar(
        accuracy_df['Configuration'],
        accuracy_df['Accuracy'],
        color=['blue', 'orange', 'green', 'red', 'purple', 'brown']
    )
    plt.title('Impact of Factors on Accuracy')
    plt.xlabel('Configuration')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()
else:
    print("No valid logs available for accuracy analysis.")

# -------------------------------------
# FID Calculation
# -------------------------------------
# Assuming you have paths for the real images
real_images_path = "path_to_real_images"  # Set path to your real images folder

# List of generator outputs or generated images for each configuration
generated_images_list = {
    'With BatchNorm': torch.randn(64, 100),  # Example latent vector size (64, 100)
    'Without BatchNorm': torch.randn(64, 100),
    'LR: 0.0001': torch.randn(64, 100),
    'LR: 0.001': torch.randn(64, 100),
    'Momentum: 0.5': torch.randn(64, 100),
    'Momentum: 0.9': torch.randn(64, 100)
}

# Define image transform for resizing and normalizing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Calculate FID scores for each configuration
fid_scores = {}

for label, generated_images in generated_images_list.items():
    fid_score = calculate_fid(real_images_path, generated_images, transform)
    fid_scores[label] = fid_score
    print(f"FID score for {label}: {fid_score:.4f}")

# Plot FID scores
plt.figure(figsize=(10, 6))
plt.bar(fid_scores.keys(), fid_scores.values(), color='skyblue')
plt.title('FID Scores for Different Configurations')
plt.xlabel('Configuration')
plt.ylabel('FID Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fid_comparison.png')
plt.show()
