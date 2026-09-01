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
        self.img.show()

    def process_img(self) -> np.ndarray:
        if self.img is None:
            raise ValueError("No image has been loaded yet.")
        array = np.array(self.img)
        return array

    def is_identical(self, comp_array, name):
        is_identical = np.array_equal(self.process_img(), comp_array)
        print(f"Is {self.name} identical to {name}", is_identical)
        if not is_identical:
            # 1. Find the coordinates where they are NOT equal
            # mismatch_indices will contain 3 arrays: (Y_coordinates, X_coordinates, Channel_coordinates)
            mismatch_indices = np.where(self.process_img() != comp_array)
            
            # 2. Count how many individual color values differ
            total_mismatches = len(mismatch_indices[0])
            print(f"Total differing values across all channels: {total_mismatches}")
            
            # 3. Print the first 5 mismatched positions as an example (Y, X, Channel)
            print("First 5 mismatch positions (Row, Col, Channel):")
            for i in range(min(5, total_mismatches)):
                y = mismatch_indices[0][i]
                x = mismatch_indices[1][i]
                c = mismatch_indices[2][i]
                
                val1 = ortho_array[y, x, c]
                val2 = overlay_array[y, x, c]
                print(f"Position (Y:{y}, X:{x}, Ch:{c}) -> Ortho: {val1}, Overlay: {val2}")


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
ortho.is_identical(overlay_array, 'overlay_array')

print('---------')
is_identical = np.array_equal(ortho_array, overlay_array)
print("Are they identical?:", is_identical)

