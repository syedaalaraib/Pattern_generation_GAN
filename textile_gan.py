import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
import pandas as pd  # Add this to the import section


# -------------------------------------
# Textile Dataset Class
# -------------------------------------
class TextileDataset(Dataset):
    """
    Custom Dataset class for loading Batik and Clothing images.
    Each class is labeled with a numeric value (Batik=0, Clothing=1).
    """
    def __init__(self, batik_dir, clothing_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load Batik images
        for root, _, files in os.walk(batik_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(0)  # Label Batik as 0
        
        # Load Clothing images
        for root, _, files in os.walk(clothing_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(1)  # Label Clothing as 1

    def __len__(self):
        # Return total number of images in the dataset
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and corresponding label at the given index
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# -------------------------------------
# Generator Variants
# -------------------------------------
class GeneratorWithBatchNorm(nn.Module):
    """
    Generator model with Batch Normalization.
    Uses a fully connected architecture to generate 128x128 RGB images.
    """
    def __init__(self, latent_dim=100):
        super(GeneratorWithBatchNorm, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),  # Add BatchNorm to stabilize training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),  # Add BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),  # Add BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 128*128*3),  # Output: Flattened 128x128 RGB image
            nn.Tanh()  # Scale outputs to [-1, 1] for compatibility with normalized images
        )
        
    def forward(self, z):
        # Generate image from latent vector z
        img = self.model(z)
        img = img.view(img.size(0), 3, 128, 128)  # Reshape to image dimensions
        return img


class ConvGenerator(nn.Module):
    """
    Convolutional Generator model.
    Uses transposed convolution layers to generate images.
    """
    def __init__(self, latent_dim=100):
        super(ConvGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),  # Latent vector expanded to 512 features
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Upsample to 256 features
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # Upsample to 128 features
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),  # Output 3-channel RGB image
            nn.Tanh()
        )

    def forward(self, z):
        # Generate image from latent vector z
        z = z.view(z.size(0), z.size(1), 1, 1)  # Reshape latent vector for convolutional layers
        img = self.model(z)
        return img


# -------------------------------------
# Discriminator Variants
# -------------------------------------
class DiscriminatorWithBatchNorm(nn.Module):
    """
    Discriminator model with Batch Normalization.
    Fully connected architecture to classify images as real or fake.
    """
    def __init__(self):
        super(DiscriminatorWithBatchNorm, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128*128*3, 512),
            nn.BatchNorm1d(512),  # Add BatchNorm to stabilize training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Add BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a probability between 0 (fake) and 1 (real)
        )
        
    def forward(self, img):
        # Flatten image and classify as real or fake
        img_flat = img.view(img.size(0), -1)  # Flatten image
        validity = self.model(img_flat)
        return validity


# -------------------------------------
# Training and Evaluation
# -------------------------------------
class GANTrainer:
    """
    GAN Trainer class for training and evaluating the Generator and Discriminator.
    Includes functionality for early stopping, checkpoints, and gradient clipping.
    """
    def __init__(self, generator, discriminator, dataset, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        self.latent_dim = config['latent_dim']
        self.epochs = config['epochs']
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
        self.adversarial_loss = nn.BCELoss()  # Binary Cross Entropy Loss for real/fake classification
        self.g_losses = []
        self.d_losses = []
        self.early_stopping_patience = config.get("early_stopping_patience", 10)  # Stop training if no improvement
        
        # Output directory for saving results
        self.output_dir = "gan_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self):
        """
        Train the GAN model for a specified number of epochs.
        Includes checkpointing and early stopping.
        """
        no_improvement = 0
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for real_imgs, _ in self.dataloader:
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                valid = torch.ones(batch_size, 1).to(self.device)  # Real labels
                fake = torch.zeros(batch_size, 1).to(self.device)  # Fake labels

                # Train Generator
                self.g_optimizer.zero_grad()
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                generated_imgs = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(generated_imgs), valid)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)  # Gradient clipping
                self.g_optimizer.step()

                # Train Discriminator
                self.d_optimizer.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(generated_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)  # Gradient clipping
                self.d_optimizer.step()

                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()

            avg_g_loss = g_loss_epoch / len(self.dataloader)
            avg_d_loss = d_loss_epoch / len(self.dataloader)

            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            print(f"Epoch [{epoch + 1}/{self.epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

            # Early stopping
            if avg_g_loss < best_loss:
                best_loss = avg_g_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

            # Checkpointing
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)

        # Save training logs
        self.save_training_logs()

    def save_checkpoint(self, epoch):
        """
        Save model checkpoints during training.
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, f'gan_checkpoint_epoch_{epoch}.pth'))

    def save_training_logs(self):
        """
        Save training losses to a CSV file for visualization and analysis.
        """
        logs = {
            "Epoch": list(range(1, len(self.g_losses) + 1)),
            "Generator Loss": self.g_losses,
            "Discriminator Loss": self.d_losses
        }
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(self.output_dir, "training_logs.csv"), index=False)

def train_and_save_logs(generator, discriminator, dataset, config, output_file):
    """
    Train a GAN model with a specific configuration and save the logs to a file.
    """
    trainer = GANTrainer(generator, discriminator, dataset, config)
    trainer.train()

    # Save logs to the specified output file
    logs = {
        "Epoch": list(range(1, len(trainer.g_losses) + 1)),
        "Generator Loss": trainer.g_losses,
        "Discriminator Loss": trainer.d_losses
    }
    df = pd.DataFrame(logs)
    df.to_csv(output_file, index=False)
    print(f"Logs saved to {output_file}")


def main():
    # Define paths to datasets
    batik_dir = "D:\\Genaiproject\\batik-nusantara-batik-indonesia-dataset"  # Replace with your Batik dataset path
    clothing_dir = "D:\\Genaiproject\\basic-pattern-women-clothing-dataset\\basicpattern_3800"  # Replace with your Clothing dataset path

    # Define dataset transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values to [-1, 1]
    ])

    # Load the dataset
    dataset = TextileDataset(batik_dir, clothing_dir, transform)
    print(f"Dataset loaded with {len(dataset)} images.")

    # Training configurations
    configs = [
        {
            "name": "logs_batchnorm",
            "generator": GeneratorWithBatchNorm(latent_dim=100),
            "discriminator": DiscriminatorWithBatchNorm(),
            "config": {
                'batch_size': 64,
                'learning_rate': 0.0002,
                'latent_dim': 100,
                'epochs': 10
            }
        },
        {
            "name": "logs_no_batchnorm",
            "generator": GeneratorWithBatchNorm(latent_dim=100),  # Without BatchNorm
            "discriminator": DiscriminatorWithBatchNorm(),  # Without BatchNorm
            "config": {
                'batch_size': 64,
                'learning_rate': 0.0002,
                'latent_dim': 100,
                'epochs': 10
            }
        },
        {
            "name": "logs_lr_0.0001",
            "generator": GeneratorWithBatchNorm(latent_dim=100),
            "discriminator": DiscriminatorWithBatchNorm(),
            "config": {
                'batch_size': 64,
                'learning_rate': 0.0001,  # Lower learning rate
                'latent_dim': 100,
                'epochs': 10
            }
        },
        {
            "name": "logs_lr_0.001",
            "generator": GeneratorWithBatchNorm(latent_dim=100),
            "discriminator": DiscriminatorWithBatchNorm(),
            "config": {
                'batch_size': 64,
                'learning_rate': 0.001,  # Higher learning rate
                'latent_dim': 100,
                'epochs': 10
            }
        },
        {
            "name": "logs_momentum_0.5",
            "generator": GeneratorWithBatchNorm(latent_dim=100),
            "discriminator": DiscriminatorWithBatchNorm(),
            "config": {
                'batch_size': 64,
                'learning_rate': 0.0002,
                'latent_dim': 100,
                'epochs': 10
            }
        },
        {
            "name": "logs_momentum_0.9",
            "generator": GeneratorWithBatchNorm(latent_dim=100),
            "discriminator": DiscriminatorWithBatchNorm(),
            "config": {
                'batch_size': 64,
                'learning_rate': 0.0002,
                'latent_dim': 100,
                'epochs': 10
            }
        }
    ]

    # Train each configuration and save logs
    for cfg in configs:
        print(f"Training configuration: {cfg['name']}")
        output_file = os.path.join("gan_outputs", f"{cfg['name']}.csv")
        train_and_save_logs(cfg['generator'], cfg['discriminator'], dataset, cfg['config'], output_file)

    print("All configurations trained and logs saved.")


if __name__ == "__main__":
    main()
