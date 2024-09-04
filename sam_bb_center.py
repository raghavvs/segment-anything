from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import cv2
import numpy as np

# initialize the model
sam = sam_model_registry["vit_h"](checkpoint="/home/rnavaratna/Documents/sam/checkpoints/sam_vit_h_4b8939.pth")

# create a mask generator using the model
mask_generator = SamAutomaticMaskGenerator(sam)

# Load the image using OpenCV
image_path = "/home/rnavaratna/Documents/sam/images/2.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# generate masks from the input image
masks = mask_generator.generate(image)

# Create a copy of the original image for drawing
result_image = image.copy()

# Sort masks by area in descending order
masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

# Draw masks, bounding boxes, and add IDs
for i, mask_data in enumerate(masks_sorted):
    mask = mask_data['segmentation']
    bbox = mask_data['bbox']
    area = mask_data['area']

    # Apply the mask
    color = np.random.randint(0, 255, 3)
    result_image[mask] = result_image[mask] * 0.5 + color * 0.5

    # Draw bounding box
    x, y, w, h = bbox
    cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)

    # Calculate center of bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Add ID (based on area rank) and center coordinates
    id_text = f"ID: {i+1}"
    center_text = f"({center_x}, {center_y})"
    cv2.putText(result_image, id_text, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(result_image, center_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.axis('off')
plt.title('Masks with Bounding Boxes and IDs')
plt.show()

# Save the result
output_filename = "/home/rnavaratna/Documents/sam/result_with_masks_and_boxes.png"
cv2.imwrite(output_filename, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
print(f"Saved result image: {output_filename}")

# Print information about each mask
for i, mask_data in enumerate(masks_sorted):
    bbox = mask_data['bbox']
    area = mask_data['area']
    center_x = bbox[0] + bbox[2] // 2
    center_y = bbox[1] + bbox[3] // 2
    print(f"Mask {i+1}: Area = {area}, Center = ({center_x}, {center_y})")