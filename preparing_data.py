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
        Gives a label for evry line in file, which consists of lines cordinates
        Blue - "нижняя бровка"
        Red - "вехняя бровка"
        Yellow - "хребет"
        Other - "Неизвестно"
        """
        diff_r = rgb_comp_all[:, 0].astype(int) - rgb_self_all[:, 0].astype(int)
        diff_g = rgb_comp_all[:, 1].astype(int) - rgb_self_all[:, 1].astype(int)
        diff_b = rgb_comp_all[:, 2].astype(int) - rgb_self_all[:, 2].astype(int)

        labels = []
        low_count, high_count, peak_count, unknown_count = 0, 0, 0, 0

        for i in range(len(y_indices)):
            #y, x = y_indices[i], x_indices[i]
            dr, dg, db = diff_r[i], diff_g[i], diff_b[i]
            
            if db > 0 and dr == 0 and dg == 0:
                labels.append(1)
                low_count+=1   
            elif dr > 0 and db == 0 and dg == 0:
                labels.append(2)
                high_count+=1
            elif dr > 0 and dg > 0 and db == 0:
                labels.append(3)
                peak_count+=1
            else:
                labels.append(0) 
                unknown_count+=1
            #line = (
                #f"Position (Y:{y}, X:{x}) -> {self.name} "
                #f"RGB: {rgb_self_all[i]} | Comp RGB: {rgb_comp_all[i]} | Тип: {label}\n"
            #)
            #lines.append(line)

        print(f"кол-во нижних бровок: {low_count},\nкол-во верхних бровок: {high_count},\nкол-во хребтов: {peak_count},\nкол-во неизвестных {unknown_count}")
        return labels

    def uniq_pixels(self, comp_array, boolean_matrix):
        if not boolean_matrix.all():
            y_indices, x_indices = np.where(~boolean_matrix)
            total_mismatches = len(y_indices)
            print(f"Total differing pixel coordinates: {total_mismatches}")

            img_self = self.array # берем уже сохраненный массив
            rgb_self_all = img_self[y_indices, x_indices]
            rgb_comp_all = comp_array[y_indices, x_indices]
            
            classified_lables = self.color_rule(y_indices, x_indices, rgb_self_all, rgb_comp_all)
            
            if len(classified_lables) == total_mismatches:
                with open("lables.txt", "w", encoding="utf-8") as file:
                    # Added a newline \n between numbers so they aren't merged on a single line
                    file.writelines([f"{str(num)}\n" for num in classified_lables])
                print("Файл lables.txt успешно обновлен!")
            else: 
                raise ValueError("Value of total missmatches is not eq to labels num.")

            return y_indices, x_indices, classified_lables

        print("Изображения идентичны. Возвращаются пустые структуры.")
        return np.array([], dtype=int), np.array([], dtype=int), []



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


class Mask():

    def __init__(self, filename):
        self.filename = filename

    def make_mask(self, axis_y, axis_x, val):
        matrix_2d = np.zeros((1024, 1024), dtype=int)

        # x = axis_x.tolist()
        # print(x)
        print(matrix_2d.shape)  # Output: (1024, 1024)
        print(matrix_2d[0][0]) 
        for i in range(len(val)):
            matrix_2d[axis_y[i]][axis_x[i]] = val[i]

        np.savetxt('final_mask.csv', matrix_2d, delimiter=',', fmt='%d')
        print("Saved as a human-readable CSV file!")

    def count_values_from_csv(self):
        matrix = np.loadtxt(self.filename, delimiter=',', dtype=int)

        unique_labels, counts = np.unique(matrix, return_counts=True)
        print(f"Loaded matrix shape: {matrix.shape}") 

        for label, count in zip(unique_labels, counts):
            label_name = "Background / Empty (0)" if label == 0 else f"Label {label}"
            if label == 1: label_name = "Нижняя бровка (1)"
            elif label == 2: label_name = "Верхняя бровка (2)"
            elif label == 3: label_name = "Хребет (3)"
            print(f"{label_name}: {count} pixels")
        

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
axis_y, axis_x, val = ortho.uniq_pixels(overlay_array, matrix_result)

different_pixels_mask = ~matrix_result
overlay_mismatched_values = overlay_array[different_pixels_mask]

print("Shape of extracted values:", overlay_mismatched_values.shape)
print("First 5 mismatched pixel values from overlay:\n", overlay_mismatched_values[:5])

print('---------')

mask = Mask('final_mask.csv')
mask.make_mask(axis_y, axis_x, val)
mask.count_values_from_csv()

