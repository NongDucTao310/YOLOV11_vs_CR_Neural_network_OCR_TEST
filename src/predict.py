from ultralytics import YOLO
import argparse
import os
import cv2
import shutil
from tkinter import Tk, messagebox

parser = argparse.ArgumentParser(description="Run YOLO predictions.")
parser.add_argument("--input_folder", required=True, help="Path to the input folder containing images.")
parser.add_argument("--output_folder", default="./Predict_Output", help="Path to the output folder.")
parser.add_argument("--model_path", default="./model/best.pt", help="Path to the YOLO model file.")
parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold for predictions.")
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
model_path = args.model_path
conf_threshold = args.conf_threshold

try:
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    results_folder = os.path.join(output_folder, "results")
    cropped_folder = os.path.join(output_folder, "Cropped_Images")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(cropped_folder, exist_ok=True)

    model = YOLO(model_path)

    def get_image_files(folder):
        image_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files

    image_paths = get_image_files(input_folder)

    def process_images_in_batches(image_paths, batch_size, model, conf_threshold):
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            for img_path in batch_paths:
                print(f"Processing image: {img_path}")  # Print the image being processed
            results = model(batch_paths, conf=conf_threshold)
            for img_path, result in zip(batch_paths, results):
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path}")

                relative_path = os.path.relpath(img_path, input_folder)
                relative_dir = os.path.dirname(relative_path)
                root_folder_name = os.path.basename(os.path.normpath(input_folder))

                result_output_dir = os.path.join(results_folder, root_folder_name, relative_dir)
                os.makedirs(result_output_dir, exist_ok=True)
                result_output_path = os.path.join(result_output_dir, os.path.basename(img_path))
                result.save(filename=result_output_path)

                cropped_output_dir = os.path.join(cropped_folder, root_folder_name, relative_dir, os.path.splitext(os.path.basename(img_path))[0])
                os.makedirs(cropped_output_dir, exist_ok=True)

                sorted_boxes = sorted(result.boxes, key=lambda box: (box.xyxy[0][1], box.xyxy[0][0]))

                for idx, box in enumerate(sorted_boxes, start=1):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_img = img[y1:y2, x1:x2]
                    cropped_output_path = os.path.join(cropped_output_dir, f"_result{idx}.jpg")
                    cv2.imwrite(cropped_output_path, cropped_img)
                    print(f" \n {cropped_output_path} saved ||  {cropped_img}  saved", cropped_output_path)

    batch_size = 10  # Adjust batch size based on available memory
    process_images_in_batches(image_paths, batch_size, model, conf_threshold)

    print(f"Predictions saved to: {results_folder}")
    print(f"Cropped images saved to: {cropped_folder}")
    messagebox.showinfo("Prediction Complete", f"Prediction completed successfully!\n\n"
                                               f"Results are saved in:\n"
                                               f"- Results folder: {results_folder}\n"
                                               f"- Cropped Images folder: {cropped_folder}")
    exit(0)

    root = Tk()
    root.withdraw()

except Exception as e:
    root = Tk()
    root.withdraw()
    messagebox.showerror("Error", f"An error occurred during prediction:\n{str(e)}")
