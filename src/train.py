import argparse
import os
import sys
import yaml
from ultralytics import YOLO
from tkinter import Tk, filedialog, messagebox

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train YOLO model.")
parser.add_argument("--yaml_file", required=True, help="Path to the dataset YAML file.")
parser.add_argument("--yolo_model", required=True, help="Path to the YOLO model file.")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
args = parser.parse_args()

# Use parsed arguments
yaml_file = args.yaml_file
yolo_model = args.yolo_model
epochs = args.epochs

def validate_dataset(data_path):
    """
    Validate the dataset paths and ensure the required files exist.
    """
    with open(data_path, 'r') as file:
        data = yaml.safe_load(file)

    train_path = os.path.join(data.get('path', ''), data.get('train', ''))
    val_path = os.path.join(data.get('path', ''), data.get('val', ''))

    # Validate training and validation paths
    if not train_path or not os.path.exists(train_path):
        raise ValueError(f"Training data path '{train_path}' is invalid or does not exist.")
    if not val_path or not os.path.exists(val_path):
        raise ValueError(f"Validation data path '{val_path}' is invalid or does not exist.")

    # Validate images and labels
    train_images = [f for f in os.listdir(train_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_labels_path = train_path.replace('images', 'labels')
    train_labels = [f for f in os.listdir(train_labels_path)] if os.path.exists(train_labels_path) else []

    if not train_images:
        raise ValueError(f"No images found in the training set at '{train_path}'.")
    if not train_labels:
        raise ValueError(f"No labels found in the training set at '{train_labels_path}'.")
    if len(train_images) != len(train_labels):
        raise ValueError(f"Mismatch between the number of images and labels in the training set. "
                         f"Images: {len(train_images)}, Labels: {len(train_labels)}")

    print("Dataset validation successful!")

def train_model(model_path, data_path, epochs, output_dir="../runs/train"):
    """
    Train the YOLO model using the specified parameters.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load the YOLO model
        model = YOLO(model_path)

        # Train the model
        results = model.train(data=data_path, epochs=epochs, project=output_dir, name="experiment")

        # Log training results
        print("Training completed successfully!")
        print(f"Results: {results}")  # Includes metrics like loss, mAP, etc.

        # Locate the best model file
        best_model_path = os.path.join(output_dir, "experiment", "weights", "best.pt")
        if os.path.exists(best_model_path):
            print(f"Training completed. Best model saved at: {best_model_path}")

            # Show file dialog to save the best.pt file
            root = Tk()
            root.withdraw()  # Hide the root window
            save_path = filedialog.asksaveasfilename(
                title="Save Trained Model",
                initialfile="best.pt",
                filetypes=[("PyTorch Model Files", "*.pt")]
            )
            if save_path:
                os.rename(best_model_path, save_path)
                print(f"Model saved to: {save_path}")
            else:
                print("Save operation canceled.")
        else:
            print("Training completed, but no best.pt file found.")
            print("Debugging: Checking the output directory structure...")
            for root, dirs, files in os.walk(output_dir):
                print(f"Directory: {root}")
                print(f"Subdirectories: {dirs}")
                print(f"Files: {files}")
    except Exception as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    try:
        validate_dataset(yaml_file)
        train_model(yolo_model, yaml_file, epochs)
    except Exception as e:
        # Display an error message box if an exception occurs
        root = Tk()
        root.withdraw()  # Hide the root window
        messagebox.showerror("Error", f"An error occurred during training:\n{str(e)}")
    finally:
        sys.exit(0)