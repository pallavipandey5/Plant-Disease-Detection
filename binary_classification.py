import torch
import torch.nn as nn
from torchvision.models import resnet18

# Load the pre-trained ResNet18 model
model = resnet18(pretrained=False)

# Modify the final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)  # Changing output to 2 for binary classification

# Load the model weights
weights = torch.load('resnet_BD.pth')

# Adjust the loaded state_dict to fit the current model
new_state_dict = model.state_dict()
for name, param in weights.items():
    if "fc" in name:  # If the name contains "fc"
        if "weight" in name:
            new_state_dict[name] = param[:2]  # Slice to match the shape
        elif "bias" in name:
            new_state_dict[name] = param[:2]  # Slice to match the shape
    else:
        new_state_dict[name] = param

# Load the modified state_dict
model.load_state_dict(new_state_dict)

# Evaluation mode
model.eval()




import os
import torch
from PIL import Image
from torchvision import datasets, transforms, models

# Load the ImageFolder dataset
data_dir = '/home/pallavi/Binary_clasi_train/Binary_clasi_train/dataset2/train'
image_dataset = datasets.ImageFolder(root=data_dir)

# Get the class names and indices
class_names = image_dataset.classes
for idx, class_name in enumerate(class_names):
    print(f"Index: {idx}, Class Label: {class_name}")

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load a pre-trained model
model = models.resnet18(pretrained=True)
# Change the number of output features to match your classification task
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))  # Change the output layer to match the number of classes

# Load the model's weights
model_weights_path = 'resnet_BD.pth'  # Provide the path to the model's saved weights
model.load_state_dict(torch.load(model_weights_path))
model.eval()

def classify_images_in_folder(folder_path, model):
    healthy_images = []
    diseased_images = []

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform the classification
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            print(f"Output: {output}")  # Add this line
            print(f"Predicted: {predicted.item()}")  # Add this line

        # Class 0: Healthy, Class 1: Diseased
        if predicted.item() == 0:
            diseased_images.append(filename)
        else:
            healthy_images.append(filename)

    print("Healthy images:")
    for img in healthy_images:
        print(img)

    print("\nDiseased images:")
    for img in diseased_images:
        print(img)

# Assuming `model` is the model loaded with weights
#classify_images_in_folder('/home/pallavi/Binary_clasi_train/Binary_clasi_train/dataset2/train/diseased', model)
classify_images_in_folder('/path/to/train/healthy', model)




