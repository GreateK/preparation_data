from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

class Img:
    def __init__(self, path):
        self.path = str(path)
        self.img = None
        self.array = None

    def img_stats(self):
        '''
        Displays info on the screen of such an image, from which
        the class istance was created.

        '''
        self.img = Image.open(self.path)
        with open("test_data/log.txt", "a", encoding="utf-8") as log:
            log.write(f"path:{self.path},\nformat:{self.img.format},\nsize:{self.img.size},\nmode:{self.img.mode}\n")
        print("Файл log.txt успешно обновлен!")
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
        if self.array is None:
            self.process_img()
        
        is_identical = np.array_equal(self.array, comp_array)
        with open("test_data/log.txt", "a", encoding="utf-8") as log:
            log.write(f"Is {self.path} identical to {name}? {is_identical}\n")

            channel_matches = (self.array == comp_array)
            pixel_matrix = channel_matches.all(axis=-1)

            log.write(f"Shape of boolean matrix: {pixel_matrix.shape}\n")  # Will be 2D: (Height, Width)
            log.write(f"Type of matrix elements: {pixel_matrix.dtype}\n")  # Will be bool

            log.write(f"Total matching pixels: {np.sum(pixel_matrix)}\n")
            log.write(f"Total different pixels: {np.sum(~pixel_matrix)}\n\n")
        return pixel_matrix

    def color_rule(self, y_indices, rgb_self_all, rgb_comp_all):
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
        with open("test_data/log.txt", "a", encoding="utf-8") as log:
            log.write(f"кол-во нижних бровок: {low_count},\nкол-во верхних бровок: {high_count},\nкол-во хребтов: {peak_count},\nкол-во неизвестных {unknown_count}")
        return labels

    def uniq_pixels(self, comp_array, boolean_matrix):
        if not boolean_matrix.all():
            y_indices, x_indices = np.where(~boolean_matrix)
            total_mismatches = len(y_indices)
            with open("test_data/log.txt", "a", encoding="utf-8") as log:
                log.write(f"Total differing pixel coordinates: {total_mismatches}")

            img_self = self.array # берем уже сохраненный массив как атрибут класса
            rgb_self_all = img_self[y_indices, x_indices]
            rgb_comp_all = comp_array[y_indices, x_indices]
            
            classified_lables = self.color_rule(y_indices, rgb_self_all, rgb_comp_all)
            
            if len(classified_lables) == total_mismatches:
                with open("test_data/lables.txt", "w", encoding="utf-8") as file:
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
        for i in range(len(val)):
            matrix_2d[axis_y[i]][axis_x[i]] = val[i]

        np.savetxt('test_data/final_mask.csv', matrix_2d, delimiter=',', fmt='%d')

    def count_values_from_csv(self):
        matrix = np.loadtxt(self.filename, delimiter=',', dtype=int)
        with open("test_data/log.txt", "a", encoding="utf-8") as log:
            log.write(f"Loaded matrix shape: {matrix.shape}\n") 

        unique_labels, counts = np.unique(matrix, return_counts=True)

        for label, count in zip(unique_labels, counts):
            label_name = "Background / Empty (0)" if label == 0 else f"Label {label}"
            if label == 1: label_name = "Нижняя бровка (1)"
            elif label == 2: label_name = "Верхняя бровка (2)"
            elif label == 3: label_name = "Хребет (3)"
            print(f"{label_name}: {count} pixels")

        basis = matrix.size 
        with open("test_data/log.txt", "a", encoding="utf-8") as log:
            log.write(f"Calculated basis (total pixels): {basis}\n")

        if basis != 1_048_576:
            raise TypeError(f"Ortho has to be 1024*1024. Actually got {basis}. Ortho name: {self.filename}")
        with open("test_data/log.txt", "a", encoding="utf-8") as log:
            log.write("Basis is correct")

    def csv_to_upscaled_colored_image(self, output_png_path='test_data/mask_hq_visual.png', upscale_factor=4):
        """
        Loads the CSV matrix, maps custom colors to labels, 
        upscales the pixel lines sharply, and displays/saves the image.
        """
        matrix = np.loadtxt(self.filename, delimiter=',', dtype=np.uint8)
        
        img = Image.fromarray(matrix, mode='P')
        
        palette = [
            0, 0, 0,          # 0: Black Background
            50, 100, 255,     # 1: Electric Blue lines
            255, 50, 50,      # 2: Bright Red lines
            255, 255, 50,     # 3: Golden Yellow
        ]
        
        # Pad palette up to 768 integers (required by Pillow for 256 possible colors)
        palette += [0] * (768 - len(palette))
        img.putpalette(palette)
        
        # Calculate new dimensions (e.g., 1024 * 4 = 4096 x 4096)
        new_width = img.width * upscale_factor
        new_height = img.height * upscale_factor
        
        # Image.NEAREST is the magic key—it multiplies the pixel size 
        # without introducing blur or fading out the narrow lines
        upscaled_img = img.resize((new_width, new_height), resample=Image.NEAREST)
        
        upscaled_img.save(output_png_path)
        with open("test_data/log.txt", "a", encoding="utf-8") as log:
            log.write(f"Upscaled colored mask ({new_width}x{new_height}) saved to {output_png_path}")
        #upscaled_img.show()
        return upscaled_img

    

def process_pipeline(input_ortho_dir_str: str, output_dir_str: str):
    input_ortho_dir = Path(input_ortho_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    ortho_files = sorted(input_ortho_dir.glob("*_ortho.png"))
    
    print(f"Found {len(ortho_files)} files to process.")

    for ortho_path in ortho_files:
        overlay_name = ortho_path.name.replace("_ortho.png", "_overlay.png")
        overlay_path = ortho_path.parent.parent / "overlays" / overlay_name

        if not overlay_path.exists():
            print(f"Skipping {ortho_path.name}: Matching overlay not found at {overlay_path}")
            continue
        with open("test_data/log.txt", "a", encoding="utf-8") as log:
            log.write(f"\nProcessing pair: {ortho_path.name} <---> {overlay_name}\n")

        try:
            ortho = Img(ortho_path)
            ortho.img_stats()
            overlay = Img(overlay_path)
            overlay.img_stats()
            overlay_array = overlay.process_img()
            #np.save('my_array.npy', ortho_array)

            matrix_result = ortho.is_identical(overlay_array, 'overlay_array')
            axis_y, axis_x, val = ortho.uniq_pixels(overlay_array, matrix_result)

            mask = Mask('test_data/final_mask.csv')
            mask.make_mask(axis_y, axis_x, val)
            mask.count_values_from_csv()

            result = mask.csv_to_upscaled_colored_image(upscale_factor=4)
            result_name = ortho_path.name.replace("_ortho.png", "_mask.png")
            result.save(output_dir/result_name)
            print(f"Successfully saved processed mask layout to: {os.getenv("OUTPUT_DIR")}")

        except Exception as e:
            print(f"Processing crashed in time of working on {ortho_path}, because of {e}")


def main():
    process_pipeline(os.getenv("ORTHO_DIR"), os.getenv("OUTPUT_DIR"))

if __name__ == "__main__":
    main()

