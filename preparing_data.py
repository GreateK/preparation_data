from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os
import re

load_dotenv()

class Img:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.img = None
        self.array = None

    def img_stats(self):
        '''
        Displays info on the screen of such an image, from which
        the class istance was created.

        '''
        self.img = Image.open(self.path)
        print('name:', self.name ,'\nformat:', self.img.format, '\nsize:', self.img.size, '\nmode:', self.img.mode)
        #self.img.show()

    def process_img(self) -> np.ndarray:
        '''
        Makes an np.array which contains pixels of such an class instance

        '''
        if self.img is None:
            raise ValueError("No image has been loaded yet.")
        self.array = np.array(self.img)
        return self.array

    def is_identical(self, comp_array, name = None):
        '''
        Compares an np.array based on such an class instance image with the other one, which 
        loaded as a parameter

        '''

        is_identical = np.array_equal(self.array, comp_array)
        print(f"Is {self.name} identical to {name}? ", is_identical)

        channel_matches = (self.array == comp_array)
        pixel_matrix = channel_matches.all(axis=-1)

        print("Shape of boolean matrix:", pixel_matrix.shape)  # Will be 2D: (Height, Width)
        print("Type of matrix elements:", pixel_matrix.dtype)  # Will be bool

        print("Total matching pixels:", np.sum(pixel_matrix))
        print("Total different pixels:", np.sum(~pixel_matrix))
        return pixel_matrix

    def color_rule(self, y_indices, x_indices, rgb_self_all, rgb_comp_all):
        """
        Применяет правила к разнице цветов и возвращает список строк для записи.
        """
        diff_r = rgb_comp_all[:, 0].astype(int) - rgb_self_all[:, 0].astype(int)
        diff_g = rgb_comp_all[:, 1].astype(int) - rgb_self_all[:, 1].astype(int)
        diff_b = rgb_comp_all[:, 2].astype(int) - rgb_self_all[:, 2].astype(int)

        lines = []
        for i in range(len(y_indices)):
            y, x = y_indices[i], x_indices[i]
            dr, dg, db = diff_r[i], diff_g[i], diff_b[i]
            
            # Определяем тип объекта по вашим правилам
            label = "Неизвестно"
            
            # 1. B > 0, R == 0, G == 0 -> нижняя бровка
            if db > 0 and dr == 0 and dg == 0:
                label = "нижняя бровка"
                
            # 2. R > 0, B == 0, G == 0 -> верхняя бровка
            elif dr > 0 and db == 0 and dg == 0:
                label = "верхняя бровка"
                
            # 3. R > 0, G > 0, B == 0 -> хребет
            elif dr > 0 and dg > 0 and db == 0:
                label = "хребет"

            # Формируем строку (если тип "Неизвестно", можно пропускать или писать как есть)
            line = (
                f"Position (Y:{y}, X:{x}) -> {self.name} "
                f"RGB: {rgb_self_all[i]} | Comp RGB: {rgb_comp_all[i]} | Тип: {label}\n"
            )
            lines.append(line)
            
        return lines

    def uniq_pixels(self, comp_array, boolean_matrix):
        if not boolean_matrix.all():
            y_indices, x_indices = np.where(~boolean_matrix)
            total_mismatches = len(y_indices)
            print(f"Total differing pixel coordinates: {total_mismatches}")

            img_self = self.array # берем уже сохраненный массив
            rgb_self_all = img_self[y_indices, x_indices]
            rgb_comp_all = comp_array[y_indices, x_indices]
            
            classified_lines = self.color_rule(y_indices, x_indices, rgb_self_all, rgb_comp_all)

            with open("differences.txt", "w", encoding="utf-8") as file:
                file.writelines(classified_lines)
            print("Файл differences.txt успешно обновлен!")


    def crop_by_differences(self, comp_array: np.ndarray, padding: int = 10):
        """
        Finds the bounding box of all differences between self and comp_array,
        crops that region out of self, and displays it.
        """
        crop = self.array
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
#np.save('my_array.npy', ortho_array)


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

print('---------')