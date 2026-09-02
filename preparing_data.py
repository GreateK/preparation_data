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
        print(f"Is {self.name} identical to {name}? ", is_identical)

        channel_matches = (self.process_img() == comp_array)
        pixel_matrix = channel_matches.all(axis=-1)

        print("Shape of boolean matrix:", pixel_matrix.shape)  # Will be 2D: (Height, Width)
        print("Type of matrix elements:", pixel_matrix.dtype)  # Will be bool

        # Quick sanity check: Count total matching pixels
        print("Total matching pixels:", np.sum(pixel_matrix))
        print("Total different pixels:", np.sum(~pixel_matrix))


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

