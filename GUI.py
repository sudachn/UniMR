import tkinter as tk
from tkinter import filedialog, Toplevel, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from scipy import spatial
import clip
import torch
import time
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

def rotate_image(image, angle):
    """旋转图像"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def inference_realesrgan(input_folder, output_folder='results', outscale=4, suffix='out', tile=0, tile_pad=10, pre_pad=0, fp32=False, ext='auto', gpu_id=None):
    # Model configuration for RealESRGAN_x4plus
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    model_name = 'RealESRGAN_x4plus'
    file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

    # Determine model path
    model_path = os.path.join('models', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = load_file_from_url(url=file_url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # Restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Process input folder or file
    if os.path.isfile(input_folder):
        paths = [input_folder]
    else:
        paths = sorted(glob.glob(os.path.join(input_folder, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if ext == 'auto':
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if suffix == '':
                save_path = os.path.join(output_folder, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(output_folder, f'{imgname}_{suffix}.{extension}')
            cv2.imwrite(save_path, output)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Start")
        self.label_instruction = tk.Label(root, text="Please enter the path containing the target molecule image and select the target molecule by outlining it.")
        self.label_instruction.pack(pady=10)
        self.frame_image_path = tk.Frame(root)
        self.frame_image_path.pack(pady=5)
        self.label_image_path = tk.Label(self.frame_image_path, text="image path：")
        self.label_image_path.pack(side=tk.LEFT, padx=(0, 10))
        self.entry = tk.Entry(self.frame_image_path, width=50)
        self.entry.pack(side=tk.LEFT)
        self.browse_button = tk.Button(root, text="Select Image Path", command=self.load_image_path)
        self.browse_button.pack()
        self.confirm_button = tk.Button(root, text="Confirm", command=self.confirm)
        self.confirm_button.pack(pady=20)
        self.molecule_info = []
        self.color_values = None
        self.threshold = 0.9
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def filter_contours_by_size(self, contours, image_shape, size_threshold_factor=2):
        """
        过滤掉过大或过小的分割区域。
        :param contours: 分割得到的轮廓列表。
        :param image_shape: 图像的形状 (height, width)。
        :param size_threshold_factor: 用于确定大小阈值的因子。默认为2，表示保留面积在平均值的2倍标准差之内的轮廓。
        :return: 过滤后的轮廓列表。
        """
        areas = [cv2.contourArea(contour) for contour in contours]
        avg_area = np.mean(areas)
        std_area = np.std(areas)
        lower_bound = max(avg_area - size_threshold_factor * std_area, 0)
        upper_bound = avg_area + size_threshold_factor * std_area

        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if lower_bound <= area <= upper_bound:
                filtered_contours.append(contour)

        return filtered_contours

    def load_image_path(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, file_path)

    def confirm(self):
        image_path = self.entry.get()
        if image_path:
            self.display_contours(image_path)

    def display_contours(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Image not found.")
            return
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.filter_contours_by_size(contours, image.shape[:2])

        self.save_contour_images(image, filtered_contours)
        self.contour_image = image.copy()
        for i, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            x = max(x - 1, 0)
            y = max(y - 1, 0)
            w = min(w + 2, image.shape[1] - x)
            h = min(h + 2, image.shape[0] - y)
            cv2.rectangle(self.contour_image, (x, y), (x+w, y+h), (255, 255, 255), 1)
            roi = image[y:y+h, x:x+w]
            self.molecule_info.append({'bbox': (x, y, w, h), 'contour': contour, 'index': i + 1, 'image': roi, 'name': None, 'color': (255, 255, 255)})

        self.image_window = Toplevel(self.root)
        self.image_window.title("Molecule Contours")

        pil_image = Image.fromarray(cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(pil_image)

        self.label = tk.Label(self.image_window, image=self.photo)
        self.label.image = self.photo
        self.label.pack()

        self.sample_button = tk.Button(self.image_window, text="Select Target Molecule", command=self.select_sample_molecule)
        self.sample_button.pack()

        tk.Label(self.image_window, text="Threshold:").pack()
        self.threshold_entry = tk.Entry(self.image_window, width=5)
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack()

        tk.Label(self.image_window, text="Max Parallel Count:").pack()
        self.max_parallel_entry = tk.Entry(self.image_window, width=5)
        self.max_parallel_entry.insert(0, "100")  # 默认值为100
        self.max_parallel_entry.pack()

        self.confirm_button = tk.Button(self.image_window, text="Confirm", command=self.classify_molecules)
        self.confirm_button.pack()
        self.image_window.update_idletasks()

    def save_contour_images(self, image, contours):
        """保存每个轮廓对应的图像区域"""
        directory = 'molecule_images'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(directory, f"molecule_{i+1}.png"), roi)

    def select_sample_molecule(self):
        def on_molecule_click(event):
            x, y = event.x, event.y
            for info in self.molecule_info:
                x1, y1, w, h = info['bbox']
                if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                    self.get_molecule_name_and_color(self.image_window)
                    if self.color_values:
                        molecule_name, r, g, b = self.color_values
                        if molecule_name and r.isdigit() and g.isdigit() and b.isdigit():
                            r, g, b = int(r), int(g), int(b)
                            info['name'] = molecule_name
                            info['color'] = (r, g, b)
                            self.mark_sample_molecule(info, (r, g, b))
                    break

        self.label.bind("<Button-1>", on_molecule_click)

    def get_molecule_name_and_color(self, parent):
        dialog = Toplevel(parent)
        dialog.title("Input Molecule Name and Color")
        
        tk.Label(dialog, text="Name:").pack()
        name_entry = tk.Entry(dialog)
        name_entry.pack()
        
        tk.Label(dialog, text="Red:").pack()
        red_entry = tk.Entry(dialog)
        red_entry.pack()
        
        tk.Label(dialog, text="Green:").pack()
        green_entry = tk.Entry(dialog)
        green_entry.pack()
        
        tk.Label(dialog, text="Blue:").pack()
        blue_entry = tk.Entry(dialog)
        blue_entry.pack()
        
        def on_submit():
            molecule_name = name_entry.get()
            r = red_entry.get()
            g = green_entry.get()
            b = blue_entry.get()
            self.color_values = (molecule_name, b, g, r)
            dialog.destroy()
        
        submit_button = tk.Button(dialog, text="Submit", command=on_submit)
        submit_button.pack()
        
        dialog.wait_window(dialog)

    def mark_sample_molecule(self, molecule_info, color):
        x, y, w, h = molecule_info['bbox']
        cv2.rectangle(self.contour_image, (x, y), (x+w, y+h), color, 1)
        pil_image = Image.fromarray(cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(pil_image)
        self.label.config(image=self.photo)

    def apply_threshold(self):
        try:
            new_threshold = float(self.threshold_entry.get())
            self.threshold = new_threshold
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for threshold.")
            return

        image = cv2.imread(self.entry.get())
        if image is None:
            print("Error: Image not found.")
            return
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def enhance_images(self, images, output_folder):
        """对图像进行增强"""
        temp_folder = 'temp_images'
        os.makedirs(temp_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        for i, image in enumerate(images):
            cv2.imwrite(os.path.join(temp_folder, f'temp_{i}.png'), image)
        inference_realesrgan(temp_folder, output_folder)
        enhanced_images = []
        for i in range(len(images)):
            enhanced_image_path = os.path.join(output_folder, f'temp_{i}_out.png')
            enhanced_image = cv2.imread(enhanced_image_path)
            if enhanced_image is None:
                raise FileNotFoundError(f"Enhanced image not found: {enhanced_image_path}")
            enhanced_images.append(enhanced_image)
        import shutil
        shutil.rmtree(temp_folder)

        return enhanced_images

    def clip_vectors(self, images):
        a = time.time()
        images = [Image.fromarray(image) for image in images]
        images = [self.preprocess(image).unsqueeze(0).to(self.device) for image in images]
        images = torch.cat(images, dim=0)
        with torch.no_grad():
            embeddings = self.model.encode_image(images)
        vectors = embeddings.cpu().numpy().tolist()
        b = time.time()
        return vectors

    def calculate_similarity(self, vector1, vector2):
        """计算两个向量之间的余弦相似度"""
        a = time.time()
        result = 1 - spatial.distance.cosine(vector1, vector2)
        b = time.time()
        return result

    def classify_molecules(self):
        try:
            self.threshold = float(self.threshold_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for threshold.")
            return
        try:
            max_parallel = int(self.max_parallel_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for max parallel count.")
            return

        sample_vectors = {}
        classified_count = {'other': 0}
        sample_images = [info['image'] for info in self.molecule_info if info.get('name')]
        enhanced_sample_images = self.enhance_images(sample_images, 'enhanced_samples')
        rotated_sample_images = []
        for sample_image in enhanced_sample_images:
            for angle in range(0, 360, 10):
                rotated_image = rotate_image(sample_image, angle)
                rotated_sample_images.append(rotated_image)
        sample_vectors_list = []
        for i in range(0, len(rotated_sample_images), max_parallel):
            batch = rotated_sample_images[i:i + max_parallel]
            sample_vectors_list.extend(self.clip_vectors(batch))
        sample_indices = [i for i, info in enumerate(self.molecule_info) if info.get('name')]
        for i, sample_index in enumerate(sample_indices):
            info = self.molecule_info[sample_index]
            for j in range(36):
                sample_vectors[(info['name'], j)] = (sample_vectors_list[i * 36 + j], info['color'])
        unknown_images = [info['image'] for info in self.molecule_info if not info.get('name')]
        enhanced_unknown_images = self.enhance_images(unknown_images, 'enhanced_unknowns')
        unknown_vectors_list = []
        for i in range(0, len(enhanced_unknown_images), max_parallel):
            batch = enhanced_unknown_images[i:i + max_parallel]
            unknown_vectors_list.extend(self.clip_vectors(batch))

        unknown_indices = [i for i, info in enumerate(self.molecule_info) if not info.get('name')]
        for i, unknown_index in enumerate(unknown_indices):
            info = self.molecule_info[unknown_index]
            molecule_vector = unknown_vectors_list[i]
            max_similarity = 0
            best_match = 'other'
            for (name, angle), (sample_vector, color) in sample_vectors.items():
                similarity = self.calculate_similarity(molecule_vector, sample_vector)
                if similarity > max_similarity and similarity > self.threshold:
                    max_similarity = similarity
                    best_match = name

            if best_match != 'other':
                info['name'] = best_match
                info['color'] = sample_vectors[(best_match, 0)][1]
                classified_count[best_match] = classified_count.get(best_match, 0) + 1
            else:
                classified_count['other'] += 1

        self.contour_image = self.contour_image.copy()
        for info in self.molecule_info:
            x, y, w, h = info['bbox']
            cv2.rectangle(self.contour_image, (x, y), (x+w, y+h), info.get('color', (255, 255, 255)), 1)

        pil_image = Image.fromarray(cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(pil_image)
        self.label.config(image=self.photo)

        result_message = "Classification Results:\n"
        for category, count in classified_count.items():
            result_message += f"{category}: {count}\n"

        result_dialog = Toplevel(self.root)
        result_dialog.title("Classification Results")

        tk.Label(result_dialog, text=result_message, wraplength=400).pack(pady=20)
        ok_button = tk.Button(result_dialog, text="OK", command=result_dialog.destroy)
        ok_button.pack()

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
