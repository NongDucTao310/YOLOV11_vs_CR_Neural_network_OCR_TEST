import cv2
import numpy as np
import matplotlib.pyplot as plt  # Thêm thư viện matplotlib

def convert_to_bw(image_path, lower_bound, upper_bound):
    """
    Convert selected color range to white and remove other colors.

    Args:
        image_path (str): Path to the input image.
        lower_bound (tuple): Lower bound of the color range in HSV (e.g., (H, S, V)).
        upper_bound (tuple): Upper bound of the color range in HSV (e.g., (H, S, V)).

    Returns:
        output_image (numpy.ndarray): The processed image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the selected color range
    mask = cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))

    # Create the output image: white for selected colors, black for others
    output_image = cv2.bitwise_and(image, image, mask=mask)
    output_image[np.where(mask > 0)] = [255, 255, 255]  # Set selected colors to white
    output_image[np.where(mask == 0)] = [0, 0, 0]      # Set other colors to black

    return output_image

if __name__ == "__main__":
    # Example usage
    input_image_path = f"C:/Users/ndt31/Desktop/DFM-SMT1-CSA/DFM-SMT1-CSA/C3808/1.3.jpg"  # Replace with your input image path

    # Define the color range in HSV for red
    lower_hsv1 = (0, 180, 100)    # Lower bound for red (first range)
    upper_hsv1 = (5, 255, 255) # Upper bound for red (first range)

    lower_hsv2 = (175, 180, 100)  # Lower bound for red (second range)
    upper_hsv2 = (180, 255, 255) # Upper bound for red (second range)

    # Convert the image
    original_image = cv2.imread(input_image_path)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv_image, np.array(lower_hsv1), np.array(upper_hsv1))
    mask2 = cv2.inRange(hsv_image, np.array(lower_hsv2), np.array(upper_hsv2))

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(original_image, original_image, mask=combined_mask)

    # Convert the masked image to grayscale
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)

    # Apply brightness thresholding
    _, processed_image = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)

    # Remove small bright spots using morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Kernel size can be adjusted
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)

    # Connect nearby bright regions using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Kernel size can be adjusted
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)

    # Convert back to 3-channel image for visualization
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

    # Update the result image for display
    result_image = processed_image

    # Convert images to RGB for matplotlib
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Plot the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Processed Image (Red Highlighted)")
    plt.imshow(result_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()