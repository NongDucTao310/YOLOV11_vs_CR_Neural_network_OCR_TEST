import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split  # Add this import

# Define a custom dataset
class OCRDataset(Dataset):
    def __init__(self, labels_file, transform=None, char_to_idx=None, unk_token="<UNK>"):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.char_to_idx = char_to_idx
        self.unk_token = unk_token

        # Load JSON file
        with open(labels_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for entry in data:
                    image_name = entry.get("image_name", "").strip()
                    ocr_value = entry.get("ocr_value", "").strip()
                    if image_name and ocr_value:
                        self.image_paths.append(image_name)
                        self.labels.append(ocr_value)
                    else:
                        print(f"Skipping invalid entry: {entry}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file: {e}")
                raise

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open("./OCR_dataset/dataset/images/"+f"{image_path}").convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)

        # Encode label into a tensor of character indices
        encoded_label = torch.tensor(
            [self.char_to_idx.get(char, self.char_to_idx[self.unk_token]) for char in label],
            dtype=torch.long
        )

        return image, encoded_label

# Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, char_to_idx, idx_to_char, blank_idx, output_model_path):
    model.train()
    best_combined_loss = float('inf')  # Initialize best combined loss
    best_model_state = None  # To store the best model state

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        printed_lines = 0  # Counter for printed lines

        # Training loop
        for images, labels in train_loader:
            images = images.to(device)
            flattened_labels = torch.cat(labels).to(device)
            label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)
            input_lengths = torch.full(size=(images.size(0),), fill_value=model.output_length, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.log_softmax(2)

            # Calculate loss
            loss = criterion(outputs, flattened_labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Decode predictions for debugging
            predicted_indices = torch.argmax(outputs, dim=2).permute(1, 0)  # Shape: (N, T)
            for i, pred in enumerate(predicted_indices):
                if printed_lines >= 50:  # Limit to 50 lines per epoch
                    break

                # Remove repeated characters and blank tokens
                cleaned_pred = []
                prev_idx = None
                for idx in pred:
                    idx = idx.item()
                    if idx != prev_idx and idx != blank_idx:
                        if idx_to_char[idx] != "<UNK>":  # Skip <UNK> tokens
                            cleaned_pred.append(idx_to_char[idx])
                    prev_idx = idx
                predicted_text = "".join(cleaned_pred)

                # Decode actual text
                actual_text = "".join([idx_to_char[idx.item()] for idx in labels[i]])

                # Print detailed information
                print(f"Epoch: {epoch + 1}, Loss: {loss.item():.8f}, Predicted: {predicted_text}, Actual: {actual_text}")
                printed_lines += 1

        # Validation loop
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                flattened_labels = torch.cat(labels).to(device)
                label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)
                input_lengths = torch.full(size=(images.size(0),), fill_value=model.output_length, dtype=torch.long).to(device)

                outputs = model(images)
                outputs = outputs.log_softmax(2)

                # Calculate validation loss
                val_loss = criterion(outputs, flattened_labels, input_lengths, label_lengths)
                total_val_loss += val_loss.item()

        # Calculate combined loss
        combined_loss = total_train_loss + total_val_loss

        # Save the best model if the combined loss is lower
        if combined_loss < best_combined_loss:
            best_combined_loss = combined_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, output_model_path)  # Save the best model immediately
            print(f"Best model updated and saved with combined loss {best_combined_loss:.8f} (Train Loss: {total_train_loss:.8f}, Val Loss: {total_val_loss:.8f}) to {output_model_path}")

        # Print losses for the epoch 
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_train_loss:.8f}, Val Loss: {total_val_loss:.8f}, Combined Loss: {combined_loss:.8f}")

        model.train()  # Switch back to training mode

# Main function
def fine_tune_crnn(labels_file, output_model_path, pretrained_model_path=None, epochs=50, batch_size=16):
    # Define character set and mapping
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()[]{}<>@#$%^&*-_+=/\\|~` "  # Full English charset
    unk_token = "<UNK>"  # Token for unknown characters
    char_to_idx = {char: idx for idx, char in enumerate(charset)}
    char_to_idx[unk_token] = len(char_to_idx)  # Add <UNK> token to the charset
    blank_idx = len(char_to_idx)  # Set blank index to the last index
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 128)),  # Resize images to a fixed size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load dataset
    dataset = OCRDataset(labels_file, transform=transform, char_to_idx=char_to_idx, unk_token=unk_token)

    # Check if the dataset is empty
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please check the labels file and ensure it contains valid data.")

    # Split dataset into training and validation sets
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the CRNN model
    class CRNN(nn.Module):
        def __init__(self, num_classes):
            super(CRNN, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.3)  # Add dropout
            )
            self.rnn = nn.LSTM(64 * 16, 128, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(128 * 2, num_classes)  # Bidirectional LSTM doubles the hidden size
            self.output_length = 32  # Example fixed output length

        def forward(self, x):
            x = self.conv(x)  # Shape: (N, C, H, W)
            x = x.permute(0, 3, 1, 2)  # Shape: (N, W, C, H)
            x = x.flatten(2)  # Shape: (N, W, C*H)
            x, _ = self.rnn(x)  # Shape: (N, W, 2*hidden_size)
            x = self.fc(x)  # Shape: (N, W, num_classes)
            return x.permute(1, 0, 2)  # Shape: (W, N, num_classes)

    model = CRNN(num_classes=len(char_to_idx) + 1)  # Add 1 for the blank token

    # Load pretrained model if provided
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}...")
        try:
            state_dict = torch.load(pretrained_model_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            print("Pretrained model loaded successfully.")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            exit(1)

    # Define loss function and optimizer
    criterion = nn.CTCLoss(blank=blank_idx)  # Use the blank index
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Add weight decay

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, char_to_idx, idx_to_char, blank_idx, output_model_path)

# Collate function for variable-length labels
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    return images, labels
    
if __name__ == "__main__":
    labels_file = r"./OCR_dataset/dataset/labels.json"  # Input labels file
    output_model_path = r"./OCR_custom_model/model2.pth"  # Output model path
    pretrained_model_path = "./OCR_custom_model/model1.pth"  # Use pre-trained model if available
    fine_tune_crnn(labels_file, output_model_path, pretrained_model_path = pretrained_model_path, epochs=100, batch_size=16)