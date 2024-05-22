import numpy as np
from PIL import Image
import cv2
import numpy as np


def image_brightness_add(image: Image.Image, brightness: int):
	"""
	Menyesuaikan kecerahan gambar dengan menambahkan nilai tertentu ke setiap piksel.
	"""
	pixels = image.load()
	for x in range(image.size[0]):
		for y in range(image.size[1]):
			# Untuk setiap r g b, tambahkan nilai brightness ke setiap piksel
			# Pastikan tidak ada nilai negatif dan melebihi 255
			pixels[x, y] = tuple([max(0, min(255, p + brightness)) for p in pixels[x, y]])
	return image

def squish_image(image: Image.Image, size: tuple[int, int]):
	width, height = size
	return image.resize((width, height), Image.LANCZOS)	

def big_head_effect(image, amount=1.5):
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    height, width, channels = img_array.shape

    # Define the center and region size
    center_x, center_y = width // 2, height // 2
    region_size = min(center_x, center_y) // 2

    # Create a meshgrid of coordinates
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate the distances from the center
    dist_x = (x_coords - center_x) / region_size
    dist_y = (y_coords - center_y) / region_size
    distances = np.sqrt(dist_x**2 + dist_y**2)

    # Apply scaling based on distance
    scale = np.where(distances < 1, 1 + (amount - 1) * (1 - distances), 1)

    # Calculate new coordinates
    new_x_coords = center_x + (x_coords - center_x) / scale
    new_y_coords = center_y + (y_coords - center_y) / scale

    # Clip coordinates to image boundaries
    new_x_coords = np.clip(new_x_coords, 0, width - 1).astype(np.float32)
    new_y_coords = np.clip(new_y_coords, 0, height - 1).astype(np.float32)

    # Map the coordinates to the original image using remap
    new_img_array = cv2.remap(img_array, new_x_coords, new_y_coords, interpolation=cv2.INTER_LINEAR)

    # Convert back to PIL Image
    output_image = Image.fromarray(new_img_array)
    return output_image

def blend_images(image1: Image.Image, image2: Image.Image, percentage: int):
	# If the size of the images are not the same, raise an error
	if image1.size != image2.size:
		image2 = squish_image(image2, image1.size)

	pixels1 = np.array(image1)
	pixels2 = np.array(image2)
	
	# Combine the images
	pixels1 = (pixels1 * (1 - percentage / 100) + pixels2 * (percentage / 100)).astype(np.uint8)

	return Image.fromarray(pixels1)
