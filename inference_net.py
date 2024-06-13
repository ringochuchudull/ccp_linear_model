from PIL import Image

import torch
import torch.optim as optim

from torchvision import transforms

from model import SimpleLinearModel


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = SimpleLinearModel(2)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def prepare_image(image_path, transform):
    image = Image.open(image_path)
    image = image.convert('RGB')  # Convert grayscale images to RGB if needed
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Load the model checkpoint
checkpoint_path = 'checkpoints/weight.pth'  # Update with your checkpoint path
model = load_checkpoint(checkpoint_path)

# Define the same transform as during training (without augmentation if any)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_paths = ['data/test/001.jpg', 'data/test/002.jpg'] # OOCL, Marsek
for ip in image_paths:
    image_tensor = prepare_image(ip, transform)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_class_index = predicted.item()
    print(f'Predicted class index: {predicted_class_index}')

