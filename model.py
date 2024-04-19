import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
import sys
import cv2
from facenet_pytorch import MTCNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Variation Encoder Class
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder and Decoder layers
        # Encoder is the same
        self.enc_conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, 7)

        # This represents the fully connected layers that output the mean and log-variance, this is new!
        self.fc_mu = nn.Linear(in_features=64*10*10, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=64*10*10, out_features=latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(in_features=latent_dim, out_features=64*10*10)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 7)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)

    # Apply reparametrization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.dec_fc(z)
        z = z.view(z.size(0), 64, 10, 10)  # Unflatten z to match the shape after the last conv layer in the encoder
        z = F.relu(self.dec_conv1(z))
        z = F.relu(self.dec_conv2(z))
        z = torch.sigmoid(self.dec_conv3(z))  # Sigmoid activation to get the output between 0 and 1
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
latent_dim = 80 
# Generator Class
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Discriminator Class
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape_flat = int(torch.prod(torch.tensor(img_shape)))

        self.model = nn.Sequential(
            nn.Linear(self.img_shape_flat, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Load the saved models
vae = VAE()
vae.load_state_dict(torch.load('vae.pth'))
vae.eval()

latent_dim = 80
img_shape = (3, 64, 64)

generator = Generator(latent_dim, img_shape)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

discriminator = Discriminator(img_shape)
discriminator.load_state_dict(torch.load('discriminator.pth'))
discriminator.eval()

# print("Models Loaded Successfully")

# Extracting frames from the given video
# Path to the input video file
def predict_video(video_path):
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True)

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)

    # List to store preprocessed faces
    face_frames = []

    # Read the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame)

        # If faces are detected, crop and preprocess each face
        if boxes is not None:
            for box in boxes:
                # Extract coordinates of the bounding box
                x, y, w, h = map(int, box)

                # Crop the face from the frame
                face = frame[y:y+h, x:x+w]

                # Preprocess the face (e.g., resize, convert to tensor, etc.)
                # Example preprocessing (resizing to 64x64 pixels)
                face = cv2.resize(face, (64, 64))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                # Append the preprocessed face to the list
                face_frames.append(face)

    # Release the video capture object
    cap.release()

    # Initialize lists to store probability scores for each frame
    frame_scores = []

    # Loop through each frame in the video
    for frame in face_frames:
        # Check the dimensions of the frame tensor
        if len(frame.shape) == 3:  # If the tensor has 3 dimensions (H x W x C)
            # Permute the dimensions to [C, H, W] and add a batch dimension
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        elif len(frame.shape) == 4:  # If the tensor already has a batch dimension
            # Permute the dimensions to [N, C, H, W]
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        else:
            raise ValueError("Unexpected number of dimensions in frame tensor")

        # Pass the frame through the VAE to extract features
        with torch.no_grad():
            features, _, _ = vae(frame_tensor)

        # Pass the extracted features through the GAN's discriminator
        with torch.no_grad():
            output = discriminator(features)
        
        # Append the probability score to the list
        frame_scores.append(output.item())

    # Aggregate the probability scores (e.g., by averaging)
    video_score = sum(frame_scores) / len(frame_scores)

    # Make a final prediction based on the aggregated score
    if video_score >= 0.5:
        return "real"
    else:
        return "fake"