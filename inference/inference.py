from torchvision import transforms
import torch
from PIL import Image

image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])

from ml.cnn.model import ConvNeuralNetwork
model = ConvNeuralNetwork()
model.load_state_dict(torch.load('models/try1.pth'))
model.eval()
image_path = 'data/nine.png'

def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image_transforms(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

    # Assuming your model predicts class indices
    return predicted.item(), confidence[predicted].item()


class_index, confidence = classify_image(image_path)
print(f"Predicted class: {class_index}, Confidence: {confidence:.2f}%")