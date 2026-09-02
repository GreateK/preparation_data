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
        height, width = boolean_matrix.shape
        is_identical = boolean_matrix.all()

        if not is_identical:
            # contains 3 arrays: (Y_coordinates, X_coordinates, Channel_coordinates)
            mismatch_indices = np.where(self.process_img() != comp_array)
            
            # gets the count of different values across all channels
            total_mismatches = len(mismatch_indices[0])
            print(f"Total differing sub-channel elements: {total_mismatches}")
            
            # passes the variable into loop range
            print("First 5 mismatch positions (Row, Col, Channel):")
            for i in range(min(5, total_mismatches)):
                y = mismatch_indices[0][i]
                x = mismatch_indices[1][i]
                c = mismatch_indices[2][i]
                
                val1 = self.process_img()[y, x, c]
                val2 = comp_array[y, x, c]
                print(f"Position (Y:{y}, X:{x}, Ch:{c}) -> {self.name}: {val1}, Comp: {val2}")



ortho = Img('ortho', os.getenv("ORTHO1"))
ortho.img_stats()
ortho_array = ortho.process_img()
print('---------')
overlay = Img('overlay', os.getenv("OVERLAY1"))
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

