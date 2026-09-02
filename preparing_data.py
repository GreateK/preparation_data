from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

class Img:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.img = None

    def img_stats(self):
        self.img = Image.open(self.path)
        print('name:', self.name ,'\nformat:', self.img.format, '\nsize:', self.img.size, '\nmode:', self.img.mode)
        #self.img.show()

    def process_img(self) -> np.ndarray:
        if self.img is None:
            raise ValueError("No image has been loaded yet.")
        array = np.array(self.img)
        return array

    def is_identical(self, comp_array, name = None):
        is_identical = np.array_equal(self.process_img(), comp_array)
        print(f"Is {self.name} identical to {name}? ", is_identical)

        channel_matches = (self.process_img() == comp_array)
        pixel_matrix = channel_matches.all(axis=-1)

        print("Shape of boolean matrix:", pixel_matrix.shape)  # Will be 2D: (Height, Width)
        print("Type of matrix elements:", pixel_matrix.dtype)  # Will be bool

        print("Total matching pixels:", np.sum(pixel_matrix))
        print("Total different pixels:", np.sum(~pixel_matrix))
        return pixel_matrix

    def uniq_pixels(self, comp_array, boolean_matrix):
        is_identical = boolean_matrix.all()

        if not is_identical:
            # 1. Use the boolean matrix to find the 2D coordinate positions (Y, X) where pixels differ
            # This avoids looping through the same pixel multiple times for different channels
            y_indices, x_indices = np.where(~boolean_matrix)
            
            total_mismatches = len(y_indices)
            print(f"Total differing pixel coordinates: {total_mismatches}")
            
            # 2. Loop through the unique pixel positions
            print("First 5 mismatch positions and their full RGB values:")
            with open("differences.txt", "w", encoding="utf-8") as file:
                for i in range(total_mismatches):
                    y = y_indices[i]
                    x = x_indices[i]
                    
                    # 3. Extract the entire RGB array at once by omitting the 'c' index
                    rgb_self = self.process_img()[y, x]
                    rgb_comp = comp_array[y, x]

                    file.write(f"Position (Y:{y}, X:{x}) -> {self.name} RGB: {rgb_self} | Comp RGB: {rgb_comp}\n")


    def crop_by_differences(self, comp_array: np.ndarray, padding: int = 10):
        """
        Finds the bounding box of all differences between self and comp_array,
        crops that region out of self, and displays it.
        """
        crop = self.process_img()
        different_pixels_mask = (crop != comp_array).any(axis=-1)
        y_indices, x_indices = np.where(different_pixels_mask)
        
        if len(y_indices) == 0:
            print(f"No differences found between {self.name} and the compared array.")
            return None

        ymin, ymax = np.min(y_indices), np.max(y_indices)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        
        # add padding so the borders of the differences aren't cut off tightly
        height, width, _ = crop.shape
        ymin = max(0, ymin - padding)
        ymax = min(height, ymax + padding)
        xmin = max(0, xmin - padding)
        xmax = min(width, xmax + padding)
        
        print(f"BBox of differences -> Y: [{ymin}:{ymax}], X: [{xmin}:{xmax}]")
        
        cropped_array = crop[ymin:ymax, xmin:xmax]
        cropped_image = Image.fromarray(cropped_array)
        cropped_image.show()
        
        return cropped_image



ortho = Img('ortho', os.getenv("ORTHO2"))
ortho.img_stats()
ortho_array = ortho.process_img()
print('---------')
overlay = Img('overlay', os.getenv("OVERLAY2"))
overlay.img_stats()
overlay_array = overlay.process_img()
np.save('my_array.npy', ortho_array)


print('---------')
print(ortho_array.shape)
print(overlay_array.shape)

print('---------')
matrix_result = ortho.is_identical(overlay_array, 'overlay_array')

print('---------')
ortho.uniq_pixels(overlay_array, matrix_result)

different_pixels_mask = ~matrix_result
overlay_mismatched_values = overlay_array[different_pixels_mask]

print("Shape of extracted values:", overlay_mismatched_values.shape)
print("First 5 mismatched pixel values from overlay:\n", overlay_mismatched_values[:5])
