import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import io

# Define the Generator class (make sure it matches the checkpoint architecture)
class GeneratorWithBatchNorm(nn.Module):
    def __init__(self, latent_dim=100):
        super(GeneratorWithBatchNorm, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 49152),  # Match the checkpoint model size
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 128, 128)  # Adjust dimensions
        return img

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = GeneratorWithBatchNorm(latent_dim=100)
checkpoint = torch.load("gan_output/checkpoints/gan_checkpoint_epoch_50.pth")
model.load_state_dict(checkpoint['generator_state_dict'], strict=False)  # Use strict=False to load matching layers
model.eval()  # Set to evaluation mode

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    # Read the image from the POST request
    img_file = request.files['image']
    img_bytes = img_file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = transform(img).unsqueeze(0)

    # Generate prediction
    with torch.no_grad():
        generated_img = model(img)

    return jsonify({"message": "Image generated successfully."})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
