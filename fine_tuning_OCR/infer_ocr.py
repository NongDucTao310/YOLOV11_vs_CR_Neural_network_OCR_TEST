import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Define the CRNN model (same as in training script)
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.rnn = nn.LSTM(64 * 16, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes)
        self.output_length = 32

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)

# Function to load the model
def load_model(model_path, char_to_idx):
    num_classes = len(char_to_idx) + 1  # Add 1 for the blank token
    model = CRNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Function to perform OCR on a single image
def ocr_image(model, image_path, transform, idx_to_char, blank_idx):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        outputs = outputs.log_softmax(2)
        predicted_indices = torch.argmax(outputs, dim=2).squeeze(1)  # Shape: (T,)
        cleaned_pred = []
        prev_idx = None
        for idx in predicted_indices:
            idx = idx.item()
            if idx != prev_idx and idx != blank_idx:
                if idx_to_char[idx] != "<UNK>":
                    cleaned_pred.append(idx_to_char[idx])
            prev_idx = idx
        return "".join(cleaned_pred)

# Function to process all images in a folder (including subfolders)
def process_folder(model, folder_path, transform, idx_to_char, blank_idx):
    results = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image_path = os.path.join(root, file)
                text = ocr_image(model, image_path, transform, idx_to_char, blank_idx)
                results[image_path] = text
    return results

if __name__ == "__main__":
    # Define character set and mapping (same as in training script)
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()[]{}<>@#$%^&*-_+=/\\|~` "
    unk_token = "<UNK>"
    char_to_idx = {char: idx for idx, char in enumerate(charset)}
    char_to_idx[unk_token] = len(char_to_idx)
    blank_idx = len(char_to_idx)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Define transformations (same as in training script)
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the trained model
    model_path = r"./model1.pth"  # Path to the trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, char_to_idx)
    print("Model loaded successfully.")

    # Folder containing images
    input_folder = r"../Predict_Output/Cropped_Images"  # Replace with your folder path
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        exit(1)
    print(f"Processing images in folder: {input_folder}...")

    # Perform OCR on all images in the folder
    results = process_folder(model, input_folder, transform, idx_to_char, blank_idx)

    # Print results
    if not results:
        print("No images found in the folder.")
    else:
        print("OCR Results:")
        for image_path, text in results.items():
            print(f"Image: {image_path}, Text: {text}")
