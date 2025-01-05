import tkinter as tk
from tkinter import filedialog, Toplevel, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from scipy import spatial
import clip
import torch


def rotate_image(image, angle):
    """旋转图像"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

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

        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.save_contour_images(image, contours)

        # 在原始图像上绘制轮廓和外接矩形
        self.contour_image = image.copy()
        for i, contour in enumerate(contours):
            # 跳过第一个轮廓，即整个图像
            if i == 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # 扩大框选范围
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

        # 在新窗口中显示图像
        self.label = tk.Label(self.image_window, image=self.photo)
        self.label.image = self.photo
        self.label.pack()

        self.sample_button = tk.Button(self.image_window, text="Select Target Molecule", command=self.select_sample_molecule)
        self.sample_button.pack()

        tk.Label(self.image_window, text="Threshold:").pack()
        self.threshold_entry = tk.Entry(self.image_window, width=5)
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack()
        self.apply_threshold_button = tk.Button(self.image_window, text="Apply Threshold", command=self.apply_threshold)
        self.apply_threshold_button.pack()

        self.confirm_button = tk.Button(self.image_window, text="Confirm", command=self.classify_molecules)
        self.confirm_button.pack()
        self.image_window.update_idletasks()
    
    def close_image_window(self):
        self.image_window.destroy()

    def save_contour_images(self, image, contours):
        directory = 'molecule_images'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, contour in enumerate(contours):
            # 跳过第一个轮廓
            if i == 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(directory, f"molecule_{i+1}.png"), roi)

    def select_sample_molecule(self):
        def on_molecule_click(event):
            # 获取点击的坐标
            x, y = event.x, event.y
            for info in self.molecule_info:
                x1, y1, w, h = info['bbox']
                if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                    # 弹出对话框让用户输入样例分子的名称和颜色
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
            self.color_values = (molecule_name, r, g, b)
            dialog.destroy()
        
        submit_button = tk.Button(dialog, text="Submit", command=on_submit)
        submit_button.pack()
        
        dialog.wait_window(dialog)

    def submit(self, dialog, on_submit):
        molecule_name, r, g, b = on_submit()
        if molecule_name and r.isdigit() and g.isdigit() and b.isdigit():
            return molecule_name, r, g, b
        else:
            messagebox.showerror("Error", "Please enter valid name and color values")
            return None, None, None, None

    def save_sample_molecule(self, info, molecule_name, color):
        contour = info['contour']
        x, y, w, h = cv2.boundingRect(contour)
        roi = self.contour_image[y:y+h, x:x+w]
        directory = 'sample_molecules'
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(os.path.join(directory, f"{molecule_name}.png"), roi)

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

    def clip_vector(self, molecule_image):
        device = "cuda:0"
        model, preprocess = clip.load(r".\models\ViT-B-32.pt")
        molecule_image = Image.fromarray(molecule_image)
        embedding = torch.flatten(preprocess(molecule_image).unsqueeze(0).to(device))
        vector = embedding.tolist()
        return vector

    def calculate_similarity(self, vector1, vector2):
        result = 1 - spatial.distance.cosine(vector1, vector2)
        return result
    
    def rotate_image(self, image, angle):
        """旋转图像"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def classify_molecules(self):
        sample_vectors = {}
        classified_count = {'other': 0}

        for info in self.molecule_info:
            if info.get('name'):
                sample_image = info['image']
                best_similarity = 0
                best_vector = None
                for angle in range(0, 360, 10):
                    rotated_image = self.rotate_image(sample_image, angle)
                    rotated_vector = self.clip_vector(rotated_image)
                    if self.calculate_similarity(rotated_vector, rotated_vector) > best_similarity:
                        best_similarity = self.calculate_similarity(rotated_vector, rotated_vector)
                        best_vector = rotated_vector
                if best_vector is not None:
                    sample_vectors[info['name']] = (best_vector, info['color'])

        for info in self.molecule_info:
            if not info.get('name'):
                molecule_vector = self.clip_vector(info['image'])
                max_similarity = 0
                best_match = 'other'
                for name, (sample_vector, color) in sample_vectors.items():
                    similarity = self.calculate_similarity(molecule_vector, sample_vector)
                    if similarity > max_similarity and similarity > self.threshold:
                        max_similarity = similarity
                        best_match = name

                if best_match != 'other':
                    info['name'] = best_match
                    info['color'] = sample_vectors[best_match][1]
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
