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

# visualize and save the masks
for i, mask_data in enumerate(masks):
    mask = mask_data['segmentation']
    # display each mask using matplotlib
    plt.imshow(mask, cmap='gray')
    plt.title(f'Mask {i+1}')
    plt.show()
    # save each mask to a file
    output_filename = f"/home/rnavaratna/Documents/sam/masks/mask_{i+1}.png"
    cv2.imwrite(output_filename, mask.astype(np.uint8) * 255)  
    print(f"Saved: {output_filename}")