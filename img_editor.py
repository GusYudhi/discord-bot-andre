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

def big_head_effect(image):
    # Convert the image to numpy array
    img_array = np.array(image)
    height, width, channels = img_array.shape

    # Define the scaling factor and the region to scale
    scale_factor = 1.5
    center_x, center_y = width // 2, height // 2
    region_size = min(center_x, center_y) // 2

    # Create an empty array for the output image
    output_array = np.zeros_like(img_array)

    for y in range(height):
        for x in range(width):
            # Calculate the distance from the center
            dist_x = (x - center_x) / region_size
            dist_y = (y - center_y) / region_size
            distance = np.sqrt(dist_x**2 + dist_y**2)

            # Scale the distance based on the scaling factor
            if distance < 1:
                scale = 1 + (scale_factor - 1) * (1 - distance)
            else:
                scale = 1

            new_x = center_x + (x - center_x) / scale
            new_y = center_y + (y - center_y) / scale

            # Ensure the new coordinates are within image boundaries
            new_x = np.clip(new_x, 0, width - 1)
            new_y = np.clip(new_y, 0, height - 1)

            output_array[y, x] = img_array[int(new_y), int(new_x)]

    # Convert the output array back to an image
    output_image = Image.fromarray(output_array)
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
