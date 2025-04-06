import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import shutil
import yaml
import argparse

parser = argparse.ArgumentParser(description="Label Tool for managing datasets.")
parser.add_argument("--image_folder", required=True, help="Path to the folder containing images.")
parser.add_argument("--dataset_folder", required=True, help="Path to the dataset folder.")
args = parser.parse_args()

image_folder = args.image_folder
dataset_folder = args.dataset_folder

def validate_folders(image_folder, dataset_folder):
    if not os.path.exists(image_folder):
        raise ValueError(f"Image folder '{image_folder}' does not exist.")
    if not os.path.exists(dataset_folder):
        raise ValueError(f"Dataset folder '{dataset_folder}' does not exist.")
    print("Folders validated successfully!")

def label_images(image_folder, dataset_folder):
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise ValueError(f"No images found in the folder '{image_folder}'.")
    
    print(f"Found {len(images)} images in '{image_folder}'.")
    print(f"Dataset folder: '{dataset_folder}'")

class LabelTool:
    def __init__(self, root, image_dir, output_dir):
        self.root = root
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.image_list = []
        self.current_image_index = 0
        self.current_image = None
        self.bbox_list = []
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.scale = 1.0
        self.rect_ids = []
        self.class_names = ["object"]
        self.selected_class = tk.StringVar(value=self.class_names[0])
        self.dataset_type = tk.StringVar(value="train")

        self.load_images()
        self.setup_ui()

    def load_images(self):
        self.image_list = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_list:
            messagebox.showerror("Error", "No images found in the specified directory!")
            self.root.quit()

    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        self.btn_save = tk.Button(self.root, text="Save", command=self.save_annotations)
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_next = tk.Button(self.root, text="Next", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_prev = tk.Button(self.root, text="Previous", command=self.previous_image)
        self.btn_prev.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_undo = tk.Button(self.root, text="Undo", command=self.undo_last_bbox)
        self.btn_undo.pack(side=tk.LEFT, padx=5, pady=5)

        self.class_menu = tk.OptionMenu(self.root, self.selected_class, *self.class_names)
        self.class_menu.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_add_class = tk.Button(self.root, text="Add Class", command=self.add_class)
        self.btn_add_class.pack(side=tk.LEFT, padx=5, pady=5)

        self.dataset_menu = tk.OptionMenu(self.root, self.dataset_type, "train", "val", "test")
        self.dataset_menu.pack(side=tk.LEFT, padx=5, pady=5)

        self.load_image()

    def load_image(self):
        image_path = os.path.join(self.image_dir, self.image_list[self.current_image_index])
        cv_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.original_image = Image.fromarray(cv_image)
        self.display_image()

    def display_image(self):
        resized_image = self.original_image.resize(
            (int(self.original_image.width * self.scale), int(self.original_image.height * self.scale))
        )
        self.current_image = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        self.root.title(f"Label Tool - {self.image_list[self.current_image_index]}")

    def on_mouse_down(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        end_x, end_y = event.x, event.y
        if end_x == self.start_x or end_y == self.start_y:
            messagebox.showwarning("Warning", "Invalid bounding box. Please draw a valid box.")
            self.canvas.delete(self.rect_id)
            return
        self.bbox_list.append((self.start_x, self.start_y, end_x, end_y))
        self.rect_ids.append(self.rect_id)
        print(f"Bounding Box: {self.start_x, self.start_y, end_x, end_y}")

    def add_class(self):
        new_class = simpledialog.askstring("Add Class", "Enter new class name:")
        if new_class and new_class not in self.class_names:
            self.class_names.append(new_class)
            self.update_class_menu()
            print(f"Class '{new_class}' added.")

    def update_class_menu(self):
        menu = self.class_menu["menu"]
        menu.delete(0, "end")
        for class_name in self.class_names:
            menu.add_command(label=class_name, command=lambda value=class_name: self.selected_class.set(value))

    def save_annotations(self):
        image_name = os.path.splitext(self.image_list[self.current_image_index])[0]
        dataset_type = self.dataset_type.get()
        folder_mapping = {"train": "train", "val": "valid", "test": "test"}
        folder_name = folder_mapping.get(dataset_type, dataset_type)
        label_output_path = os.path.join(self.output_dir, folder_name, "labels", f"{image_name}.txt")
        image_output_path = os.path.join(self.output_dir, folder_name, "images", f"{image_name}.jpg")
        image_path = os.path.join(self.image_dir, self.image_list[self.current_image_index])
        self.create_yolo_directories()
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        new_width = 640
        aspect_ratio = img_height / img_width
        new_height = int(new_width * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, new_height))
        with open(label_output_path, "w") as f:
            for bbox in self.bbox_list:
                x_min, y_min, x_max, y_max = bbox
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                class_id = self.class_names.index(self.selected_class.get())
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        cv2.imwrite(image_output_path, resized_img)
        messagebox.showinfo("Info", f"Annotations saved to {label_output_path}")
        self.generate_yaml()

    def create_yolo_directories(self):
        for subset in ["train", "valid", "test"]:
            for folder in ["images", "labels"]:
                path = os.path.join(self.output_dir, subset, folder)
                os.makedirs(path, exist_ok=True)
                print(f"Directory created or already exists: {path}")

    def generate_yaml(self):
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        yaml_content = {
            'path': self.output_dir,
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(
                yaml_content,
                yaml_file,
                default_flow_style=False,
                sort_keys=False
            )
        print(f"Dataset YAML file created at {yaml_path}")

    def next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.bbox_list = []
            self.load_image()

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.bbox_list = []
            self.load_image()

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.scale = min(self.scale * 1.1, 10)
        elif event.delta < 0:
            self.scale = max(self.scale / 1.1, 0.1)
        self.display_image()

    def undo_last_bbox(self):
        if self.bbox_list and self.rect_ids:
            self.bbox_list.pop()
            last_rect_id = self.rect_ids.pop()
            self.canvas.delete(last_rect_id)
            print("Last bounding box removed.")
        else:
            messagebox.showinfo("Info", "No bounding boxes to undo.")

if __name__ == "__main__":
    try:
        validate_folders(image_folder, dataset_folder)
        root = tk.Tk()
        app = LabelTool(root, image_folder, dataset_folder)
        root.mainloop()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred during labeling:\n{str(e)}")
