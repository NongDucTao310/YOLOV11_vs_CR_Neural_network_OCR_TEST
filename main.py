import os
import subprocess
from tkinter import Tk, Label, Button, Entry, filedialog, Toplevel, messagebox, simpledialog

process = None 

def run_script(script_path, *args):
    global process
    script_dir = os.path.dirname(script_path)  
    command = ["python", script_path] + list(args)  
    process = subprocess.Popen(command, cwd=script_dir)  

def stop_script():
    global process
    if process and process.poll() is None:  
        process.terminate()  
        process = None

def select_folder(entry):
    folder_path = filedialog.askdirectory(title="Select Folder")
    if folder_path:
        entry.delete(0, "end")
        entry.insert(0, folder_path)

def select_file(entry, filetypes):
    file_path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
    if file_path:
        entry.delete(0, "end")
        entry.insert(0, file_path)

def create_window(title, script_path):
    window = Toplevel()
    window.title(title)

    Label(window, text=f"{title} Script", font=("Arial", 16)).pack(pady=10)

    Label(window, text="Input Folder:").pack(anchor="w", padx=10)
    input_folder_entry = Entry(window, width=50)
    input_folder_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_folder(input_folder_entry)).pack(pady=5)

    Label(window, text="Output Folder (default: ./Predict_Output):").pack(anchor="w", padx=10)
    output_folder_entry = Entry(window, width=50)
    output_folder_entry.insert(0, "../Predict_Output")
    output_folder_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_folder(output_folder_entry)).pack(pady=5)

    Label(window, text="Model Path (default: ./Yolo_model/best.pt):").pack(anchor="w", padx=10)
    model_path_entry = Entry(window, width=50)
    model_path_entry.insert(0, "../Yolo_model/best.pt")
    model_path_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_file(model_path_entry, [("PyTorch Model Files", "*.pt")])).pack(pady=5)

    Label(window, text="Confidence Threshold:").pack(anchor="w", padx=10)
    conf_slider = Entry(window, width=10)
    conf_slider.pack(pady=5, padx=10)
    conf_slider.insert(0, "0.25")

    run_button = Button(window, text="Run", width=10)
    run_button.pack(pady=10)

    def toggle_run_stop():
        global process
        if run_button["text"] == "Run":
            run_script(
                script_path,
                "--input_folder", input_folder_entry.get(),
                "--output_folder", output_folder_entry.get(),
                "--model_path", model_path_entry.get(),
                "--conf_threshold", conf_slider.get()
            )
            run_button.config(text="Stop", command=toggle_run_stop)
        else:
            stop_script()
            run_button.config(text="Run", command=toggle_run_stop)

    run_button.config(command=toggle_run_stop)

def create_train_window(script_path):
    window = Toplevel()
    window.title("Train Script")

    Label(window, text="Train Script", font=("Arial", 16)).pack(pady=10)

    Label(window, text="YAML File Path:").pack(anchor="w", padx=10)
    yaml_file_entry = Entry(window, width=50)
    yaml_file_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_file(yaml_file_entry, [("YAML Files", "*.yaml")])).pack(pady=5)

    Label(window, text="YOLO Model Path:").pack(anchor="w", padx=10)
    yolo_model_entry = Entry(window, width=50)
    yolo_model_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_file(yolo_model_entry, [("PyTorch Model Files", "*.pt")])).pack(pady=5)

    Label(window, text="Epochs (default: 50):").pack(anchor="w", padx=10)
    epochs_entry = Entry(window, width=10)
    epochs_entry.insert(0, "50")
    epochs_entry.pack(pady=5, padx=10)

    run_button = Button(window, text="Run", width=10)
    run_button.pack(pady=10)

    def toggle_run_stop():
        global process
        if run_button["text"] == "Run":
            run_script(
                script_path,
                "--yaml_file", yaml_file_entry.get(),
                "--yolo_model", yolo_model_entry.get(),
                "--epochs", epochs_entry.get()
            )
            run_button.config(text="Stop", command=toggle_run_stop)
        else:
            stop_script()
            run_button.config(text="Run", command=toggle_run_stop)

    run_button.config(command=toggle_run_stop)

    def on_training_complete():
        best_model_path = os.path.join("runs", "weights", "best.pt")
        if os.path.exists(best_model_path):
            filedialog.asksaveasfilename(
                title="Save Trained Model",
                initialfile="best.pt",
                filetypes=[("PyTorch Model Files", "*.pt")]
            )
        else:
            print("Training completed, but no best.pt file found.")

    window.protocol("WM_DELETE_WINDOW", lambda: (stop_script(), window.destroy()))

def create_label_tool_window(script_path):
    window = Toplevel()
    window.title("Label Tool")

    Label(window, text="Label Tool", font=("Arial", 16)).pack(pady=10)

    Label(window, text="Image Folder:").pack(anchor="w", padx=10)
    image_folder_entry = Entry(window, width=50)
    image_folder_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_folder(image_folder_entry)).pack(pady=5)

    Label(window, text="Dataset Folder:").pack(anchor="w", padx=10)
    dataset_folder_entry = Entry(window, width=50)
    dataset_folder_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_folder(dataset_folder_entry)).pack(pady=5)

    run_button = Button(window, text="Run", width=10)
    run_button.pack(pady=10)

    def toggle_run_stop():
        global process
        if run_button["text"] == "Run":
            run_script(
                script_path,
                "--image_folder", image_folder_entry.get(),
                "--dataset_folder", dataset_folder_entry.get()
            )
            run_button.config(text="Stop", command=toggle_run_stop)
        else:
            stop_script()
            run_button.config(text="Run", command=toggle_run_stop)

    run_button.config(command=toggle_run_stop)

def run_export_to_excel_script():
    try:
        subprocess.run(["python", "./src/export_to_excel.py"], check=True)
        messagebox.showinfo("Success", "Export to Excel script executed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to run export_to_excel.py: {e}")

def open_ocr_fine_tuning():
    window = Toplevel()
    window.title("OCR Fine Tuning")

    Label(window, text="OCR Fine Tuning", font=("Arial", 16)).pack(pady=10)

    Label(window, text="Labels File:").pack(anchor="w", padx=10)
    labels_file_entry = Entry(window, width=50)
    labels_file_entry.insert(0, "./OCR_dataset/dataset/labels.json")
    labels_file_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_file(labels_file_entry, [("JSON Files", "*.json")])).pack(pady=5)

    Label(window, text="Output Model Path:").pack(anchor="w", padx=10)
    output_model_entry = Entry(window, width=50)
    output_model_entry.insert(0, "./OCR_custom_model/model2.pth")
    output_model_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_file(output_model_entry, [("PyTorch Model Files", "*.pth")])).pack(pady=5)

    Label(window, text="Pretrained Model Path (Optional):").pack(anchor="w", padx=10)
    pretrained_model_entry = Entry(window, width=50)
    pretrained_model_entry.insert(0, "./OCR_custom_model/model1.pth")
    pretrained_model_entry.pack(pady=5, padx=10)
    Button(window, text="Browse", command=lambda: select_file(pretrained_model_entry, [("PyTorch Model Files", "*.pth")])).pack(pady=5)

    Label(window, text="Epochs (default: 100):").pack(anchor="w", padx=10)
    epochs_entry = Entry(window, width=10)
    epochs_entry.insert(0, "100")
    epochs_entry.pack(pady=5, padx=10)

    Label(window, text="Batch Size (default: 16):").pack(anchor="w", padx=10)
    batch_size_entry = Entry(window, width=10)
    batch_size_entry.insert(0, "16")
    batch_size_entry.pack(pady=5, padx=10)

    def run_fine_tuning():
        labels_file = labels_file_entry.get()
        output_model_path = output_model_entry.get()
        pretrained_model_path = pretrained_model_entry.get() or None
        epochs = epochs_entry.get()
        batch_size = batch_size_entry.get()

        try:
            subprocess.run(
                ["python", "./src/OCR_fine_tuning.py", labels_file, output_model_path, pretrained_model_path, epochs, batch_size],
                check=True,
                cwd="./"  # Set working directory to ./
            )
            messagebox.showinfo("Success", "OCR Fine Tuning script executed successfully.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to run OCR_fine_tuning.py: {e}")

    Button(window, text="Run", command=run_fine_tuning).pack(pady=10)

root = Tk()
root.title("Main Application")

Label(root, text="Select and Open Script Windows", font=("Arial", 16)).pack(pady=10)

Button(root, text="Open Predict Window", command=lambda: create_window("Predict", os.path.abspath("src/predict.py"))).pack(pady=5)
Button(root, text="Open Train Window", command=lambda: create_train_window(os.path.abspath("src/train.py"))).pack(pady=5)
Button(root, text="Open Label Tool Window", command=lambda: create_label_tool_window(os.path.abspath("src/label_tool.py"))).pack(pady=5)
Button(root, text="Run Export to Excel Script", command=run_export_to_excel_script).pack(pady=5)
Button(root, text="Open OCR Fine Tuning", command=open_ocr_fine_tuning).pack(pady=5)

root.mainloop()
