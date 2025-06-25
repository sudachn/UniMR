import tkinter as tk
from tkinter import filedialog, Toplevel, simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from scipy import spatial
import clip
import torch
import glob
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
        
        # 设置模型路径
        self.model_paths = {
            'clip': 'models'  # CLIP模型路径
        }
        
        self.label_instruction = tk.Label(root, text="Please enter the path containing the target molecule image and select the target molecule by outlining it.")
        self.label_instruction.pack(pady=10)
        self.frame_image_path = tk.Frame(root)
        self.frame_image_path.pack(pady=5)
        self.label_image_path = tk.Label(self.frame_image_path, text="Image path：")
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
        
        # 初始化模型
        self.init_models()
        
    def init_models(self):
        """初始化CLIP模型"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = None
        self.preprocess = None
        try:
            if not os.path.exists(self.model_paths['clip']):
                print(f"CLIP model not found at {self.model_paths['clip']}")
                return False
            print("Loading CLIP model...")
            self.model, self.preprocess = clip.load("ViT-B/32", 
                                                  device=self.device, 
                                                  download_root=os.path.dirname(self.model_paths['clip']))
            print("CLIP model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            return False
            
    def filter_contours_by_size(self, contours, image_shape, size_threshold_factor=2, min_area=100, max_area=None):
        """
        过滤掉过大或过小的分割区域。
        :param contours: 分割得到的轮廓列表。
        :param image_shape: 图像的形状 (height, width)。
        :param size_threshold_factor: 用于确定大小阈值的因子。默认为2，表示保留面积在平均值的2倍标准差之内的轮廓。
        :param min_area: 最小面积阈值，小于此面积的轮廓将被过滤掉。
        :param max_area: 最大面积阈值，大于此面积的轮廓将被过滤掉。如果为None，则使用统计方法计算。
        :return: 过滤后的轮廓列表。
        """
        if not contours:
            return []
            
        areas = [cv2.contourArea(contour) for contour in contours]
        
        # 如果轮廓数量太少，直接返回所有轮廓
        if len(areas) <= 2:
            return contours
            
        avg_area = np.mean(areas)
        std_area = np.std(areas)
        
        # 如果没有指定最大面积，则使用统计方法计算
        if max_area is None:
            lower_bound = max(avg_area - size_threshold_factor * std_area, min_area)
            upper_bound = avg_area + size_threshold_factor * std_area
        else:
            lower_bound = min_area
            upper_bound = max_area

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

        # 缩放图像以适应固定窗口大小
        max_display_size = 600  # 设置最大显示尺寸
        height, width = image.shape[:2]
        scale_factor = min(max_display_size / width, max_display_size / height)
        display_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

        # 保存缩放比例和原始图像
        self.scale_factor = scale_factor
        self.original_image = display_image.copy()
        self.original_gray = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
        
        # 计算图像总面积
        self.total_area = self.original_image.shape[0] * self.original_image.shape[1]

        # 自动计算OTSU阈值
        otsu_value, otsu_thresh = cv2.threshold(self.original_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 确保阈值图像是正确的格式和类型
        otsu_thresh = np.array(otsu_thresh, dtype=np.uint8)
        
        # 检查图像是否有效
        if otsu_thresh is None or otsu_thresh.size == 0:
            print("Error: Invalid threshold image")
            return
            
        # 打印图像信息以便调试
        print(f"Threshold image shape: {otsu_thresh.shape}")
        print(f"Threshold image type: {otsu_thresh.dtype}")
        
        contours, hierarchy = cv2.findContours(otsu_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        otsu_areas = [cv2.contourArea(contour) for contour in contours]
        avg_area = np.mean(otsu_areas)
        std_area = np.std(otsu_areas)
        self.size_factor = 2  # 固定size_factor为2
        min_area = max(avg_area - self.size_factor * std_area, 10)
        max_area = avg_area + self.size_factor * std_area

        # 创建调整窗口
        self.adjust_window = tk.Toplevel(self.root)
        self.adjust_window.title("Adjust Parameters")
        self.adjust_window.geometry("1000x600+100+100")

        # 创建主框架
        main_frame = tk.Frame(self.adjust_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建预览图像框
        preview_frame = tk.Frame(main_frame)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # 创建控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # 阈值调整
        tk.Label(control_frame, text="Threshold:").pack(pady=(0, 5))
        self.threshold_scale = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                      length=200, command=self.update_preview)
        self.threshold_scale.set(otsu_value)  # 使用计算得到的OTSU阈值
        self.threshold_scale.pack(pady=(0, 10))

        # 最小面积调整（使用百分比）
        tk.Label(control_frame, text="Min Area (%):").pack(pady=(0, 5))
        self.min_area_scale = tk.Scale(control_frame, from_=0, to=0.1, orient=tk.HORIZONTAL, 
                                     length=200, resolution=0.001, command=self.update_preview)
        initial_min_percent = (min_area / self.total_area) * 100
        self.min_area_scale.set(min(initial_min_percent, 0.1))  # 确保不超过最大值0.1%
        self.min_area_scale.pack(pady=(0, 10))

        # 最大面积调整（使用百分比）
        tk.Label(control_frame, text="Max Area (%):").pack(pady=(0, 5))
        self.max_area_scale = tk.Scale(control_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                                     length=200, resolution=0.001, command=self.update_preview)
        initial_max_percent = (max_area / self.total_area) * 100
        self.max_area_scale.set(min(initial_max_percent, 1))  # 确保不超过最大值1%
        self.max_area_scale.pack(pady=(0, 10))

        # 应用按钮
        self.apply_button = tk.Button(control_frame, text="Apply Settings", command=self.apply_settings)
        self.apply_button.pack(pady=20)

        # 创建预览标签
        self.preview_label = tk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # 显示初始预览
        self.update_preview()

        # 等待用户确认
        self.adjust_window.wait_window()

        # 处理最终结果
        if hasattr(self, 'final_contours'):
            self.save_contour_images(display_image, self.final_contours)
            self.contour_image = display_image.copy()
            self.molecule_info = []  # 清空之前的分子信息
            
            for i, contour in enumerate(self.final_contours):
                x, y, w, h = cv2.boundingRect(contour)
                x = max(x - 1, 0)
                y = max(y - 1, 0)
                w = min(w + 2, display_image.shape[1] - x)
                h = min(h + 2, display_image.shape[0] - y)
                cv2.rectangle(self.contour_image, (x, y), (x+w, y+h), (255, 255, 255), 1)
                roi = display_image[y:y+h, x:x+w]
                self.molecule_info.append({'bbox': (x, y, w, h), 'contour': contour, 'index': i + 1, 
                                         'image': roi, 'name': None, 'color': (255, 255, 255)})

            # 显示最终结果窗口
            self.image_window = tk.Toplevel(self.root)
            self.image_window.title("Molecule Contours")
            self.image_window.geometry("1200x800+300+100")  # 增加窗口宽度以适应两栏布局

            # 创建主框架
            main_frame = tk.Frame(self.image_window)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # 左侧图像区域
            left_frame = tk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 显示图像
            pil_image = Image.fromarray(cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(pil_image)
            self.label = tk.Label(left_frame, image=self.photo)
            self.label.image = self.photo
            self.label.pack()

            # 右侧控制面板
            right_frame = tk.Frame(main_frame)
            right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

            # 在右侧创建两个子框架
            right_top_frame = tk.Frame(right_frame)
            right_top_frame.pack(fill=tk.X, pady=(0, 10))
            right_bottom_frame = tk.Frame(right_frame)
            right_bottom_frame.pack(fill=tk.X)

            # 选择目标分子按钮（顶部）
            self.sample_button = tk.Button(right_top_frame, text="Select Target Molecule", 
                                         command=self.select_sample_molecule,
                                         width=20)
            self.sample_button.pack(pady=5)

            # 阈值设置框架（左列）
            threshold_frame = tk.LabelFrame(right_bottom_frame, text="Threshold Settings", padx=10, pady=5)
            threshold_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

            # 阈值输入
            tk.Label(threshold_frame, text="Manual Threshold:").pack()
            self.threshold_entry = tk.Entry(threshold_frame, width=10)
            self.threshold_entry.insert(0, str(self.threshold))
            self.threshold_entry.pack()

            # 阈值优化方法选择
            tk.Label(threshold_frame, text="Threshold Method:").pack(pady=(10, 0))
            self.threshold_method = tk.StringVar(value="manual")
            methods = [
                ("Manual", "manual"),
                ("Otsu", "otsu"),
                ("GMM", "gmm"),
                ("Distribution", "distribution")
            ]
            for text, value in methods:
                tk.Radiobutton(threshold_frame, text=text, value=value, 
                             variable=self.threshold_method).pack()

            # 编码方法设置框架（右列）
            encoding_frame = tk.LabelFrame(right_bottom_frame, text="Encoding Settings", padx=10, pady=5)
            encoding_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

            # 编码方法选择
            self.encoding_method = tk.StringVar(value="clip")
            encodings = [
                ("CLIP Encoding", "clip"),
                ("Raw Features", "raw"),
                ("Auto Select", "auto")
            ]
            for text, value in encodings:
                tk.Radiobutton(encoding_frame, text=text, value=value, 
                             variable=self.encoding_method).pack(pady=2)

            # 并行处理设置
            tk.Label(encoding_frame, text="Max Parallel:").pack(pady=(10, 0))
            self.max_parallel_entry = tk.Entry(encoding_frame, width=10)
            self.max_parallel_entry.insert(0, "100")
            self.max_parallel_entry.pack()

            # 确认按钮（底部）
            self.confirm_button = tk.Button(right_frame, text="Start Classification", 
                                          command=self.classify_molecules,
                                          width=20)
            self.confirm_button.pack(pady=20)
            
            self.image_window.update_idletasks()

    def update_preview(self, *args):
        """更新预览图像"""
        # 获取当前参数值
        thresh_value = self.threshold_scale.get()
        # 将百分比转换为实际面积
        min_area_percent = self.min_area_scale.get()
        max_area_percent = self.max_area_scale.get()
        min_area = int((min_area_percent / 100) * self.total_area)
        max_area = int((max_area_percent / 100) * self.total_area)
        
        # 应用阈值
        _, thresh = cv2.threshold(self.original_gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
        thresh = np.uint8(thresh)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建预览图像
        preview_img = self.original_image.copy()
        
        # 在预览图像上绘制符合面积要求的轮廓
        valid_contours = []
        if min_area < max_area:  # 只在面积范围有效时处理轮廓
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    valid_contours.append(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    x = max(x - 1, 0)
                    y = max(y - 1, 0)
                    w = min(w + 2, preview_img.shape[1] - x)
                    h = min(h + 2, preview_img.shape[0] - y)
                    cv2.rectangle(preview_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 转换为PIL图像并显示
        pil_image = Image.fromarray(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB))
        self.preview_photo = ImageTk.PhotoImage(pil_image)
        self.preview_label.config(image=self.preview_photo)
        
        # 保存当前轮廓
        self.current_contours = valid_contours

    def apply_settings(self):
        """应用当前设置并关闭调整窗口"""
        self.final_contours = self.current_contours
        self.adjust_window.destroy()

    def save_contour_images(self, image, contours):
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
                    # 获取用户输入的分子名称和颜色
                    self.get_molecule_name_and_color(self.image_window)
                    if self.color_values:
                        molecule_name, r, g, b = self.color_values
                        if molecule_name and r.isdigit() and g.isdigit() and b.isdigit():
                            r, g, b = int(r), int(g), int(b)
                            info['name'] = molecule_name
                            info['color'] = (r, g, b)

                            # 将临时图像上的坐标映射回原图
                            original_x = int(x1 / self.scale_factor)
                            original_y = int(y1 / self.scale_factor)
                            original_w = int(w / self.scale_factor)
                            original_h = int(h / self.scale_factor)
                            info['original_bbox'] = (original_x, original_y, original_w, original_h)

                            # 在原图上标记选择的分子
                            self.mark_sample_molecule(info, (r, g, b))
                    break

        self.label.bind("<Button-1>", on_molecule_click)

    def get_molecule_name_and_color(self, parent):
        dialog = Toplevel(parent)
        dialog.title("Input Molecule Name and Color")
        dialog.geometry("350x250+150+350")
        
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

    def clip_vectors(self, images):
        """计算图像的CLIP向量"""
        # 检查模型是否正确加载
        if self.model is None or self.preprocess is None:
            print("CLIP model not properly initialized. Trying to reinitialize...")
            if not self.init_models():
                print("Failed to initialize CLIP model")
                return None
        
        try:
            images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
            images = [self.preprocess(image).unsqueeze(0).to(self.device) for image in images]
            images = torch.cat(images, dim=0)
            
            with torch.no_grad():
                embeddings = self.model.encode_image(images)
            
            vectors = embeddings.cpu().numpy().tolist()
            print(f"Successfully computed vectors. Vector dimension: {len(vectors[0])}")
            return vectors
            
        except Exception as e:
            print(f"Error computing CLIP vectors: {e}")
            return None

    def calculate_similarity(self, vector1, vector2):
        result = 1 - spatial.distance.cosine(vector1, vector2)
        return result

    def calculate_threshold_otsu(self, similarities):
        """Otsu's method for threshold selection (same as script)"""
        hist, bins = np.histogram(similarities, bins=256, range=(0, 1))
        hist = hist.astype(float)
        total = hist.sum()
        cumsum = np.cumsum(hist)
        cumsum2 = np.cumsum(hist * np.arange(256))
        variance = np.zeros(256)
        for t in range(256):
            if cumsum[t] == 0 or cumsum[t] == total:
                continue
            w0 = cumsum[t] / total
            w1 = 1 - w0
            mu0 = cumsum2[t] / cumsum[t]
            mu1 = (cumsum2[255] - cumsum2[t]) / (total - cumsum[t])
            variance[t] = w0 * w1 * (mu0 - mu1) ** 2
        threshold = np.argmax(variance) / 255.0
        return threshold

    def calculate_threshold_gmm(self, similarities):
        """GMM method for threshold selection (same as script)"""
        similarities_2d = similarities.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(similarities_2d)
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_
        def gaussian(x, mu, sigma, w):
            return w * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        x = np.linspace(min(similarities), max(similarities), 1000)
        y1 = gaussian(x, means[0], stds[0], weights[0])
        y2 = gaussian(x, means[1], stds[1], weights[1])
        intersection_points = []
        for i in range(len(x) - 1):
            if (y1[i] - y2[i]) * (y1[i + 1] - y2[i + 1]) <= 0:
                intersection_points.append(x[i])
        if not intersection_points:
            return np.mean(similarities) + np.std(similarities)
        return max(intersection_points)

    def calculate_threshold_distribution(self, similarities):
        """Distribution-based method for threshold selection (same as script)"""
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        threshold = mean_sim + std_sim
        ratio_above = np.mean(similarities > threshold)
        ratio_below = 1 - ratio_above
        var_above = np.var(similarities[similarities > threshold]) if ratio_above > 0 else 0
        var_below = np.var(similarities[similarities <= threshold]) if ratio_below > 0 else 0
        balance_score = 1 - abs(ratio_above - 0.5)
        variance_score = 1 / (1 + var_above + var_below)
        if balance_score < 0.3 or variance_score < 0.3:
            threshold = mean_sim + 0.5 * std_sim
        return threshold

    def select_encoding_method(self, sample_images):
        """自动选择最优的编码方法"""
        print("\nAnalyzing encoding methods...")
        
        # 获取两种编码方式的结果
        with torch.no_grad():
            # CLIP编码
            clip_embeddings = self.model.encode_image(sample_images).cpu().numpy()
            
            # 原始特征
            raw_embeddings = sample_images.cpu().numpy()
            raw_embeddings = raw_embeddings.reshape(raw_embeddings.shape[0], -1)
            
            # 计算每种方法的评分
            clip_score = self.calculate_encoding_score(clip_embeddings)
            raw_score = self.calculate_encoding_score(raw_embeddings)
            
            # 选择得分更高的方法
            if clip_score > raw_score:
                print("Automatic selection of CLIP encoding method")
                return "clip"
            else:
                print("Automatic selection of raw feature encoding method")
                return "raw"

    def calculate_encoding_score(self, embeddings):
        """计算编码质量得分"""
        # 使用PCA降维以减少计算量
        pca = PCA(n_components=min(embeddings.shape[0]-1, 50))
        embeddings_reduced = pca.fit_transform(embeddings)
        
        # 计算类内紧密度
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(embeddings_reduced)
        centers = kmeans.cluster_centers_
        
        # 计算样本到聚类中心的距离
        distances = np.zeros(len(embeddings_reduced))
        for i in range(len(embeddings_reduced)):
            distances[i] = np.linalg.norm(embeddings_reduced[i] - centers[labels[i]])
        
        # 计算评分指标
        intra_sim = 1 - np.mean(distances) / np.max(distances)
        inter_diff = np.linalg.norm(centers[0] - centers[1])
        balance = 1 - abs(np.sum(labels == 0) - np.sum(labels == 1)) / len(labels)
        
        # 综合得分
        score = (intra_sim * 0.4 + inter_diff * 0.4 + balance * 0.2)
        return score

    def classify_molecules(self):
        try:
            # 获取阈值设置方法
            threshold_method = self.threshold_method.get()
            if threshold_method == "manual":
                try:
                    self.threshold = float(self.threshold_entry.get())
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid threshold value")
                    return
            
            # 获取并行处理数量
            try:
                max_parallel = int(self.max_parallel_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for max parallel processing")
                return

            # 获取编码方法
            encoding_method = self.encoding_method.get()
            
            sample_vectors = {}
            classified_count = {'other': 0}
            
            # 获取样本图像
            sample_images = [info['image'] for info in self.molecule_info if info.get('name')]
            if not sample_images:
                messagebox.showerror("Error", "Please select at least one target molecule")
                return
                
            # 生成旋转后的样本图像
            rotated_sample_images = []
            for sample_image in sample_images:
                for angle in range(0, 360, 10):
                    rotated_image = rotate_image(sample_image, angle)
                    rotated_sample_images.append(rotated_image)
            
            # 如果是自动选择编码方法
            if encoding_method == "auto":
                # 准备样本图像用于编码方法选择
                sample_tensor = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).to(self.device) 
                                          for img in sample_images])
                encoding_method = self.select_encoding_method(sample_tensor)
            
            # 处理样本图像
            sample_vectors_list = []
            for i in range(0, len(rotated_sample_images), max_parallel):
                batch = rotated_sample_images[i:i + max_parallel]
                if encoding_method == "raw":
                    # 使用原始特征
                    processed_batch = [self.preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).to(self.device) 
                                    for img in batch]
                    processed_batch = torch.stack(processed_batch)
                    vectors = processed_batch.reshape(processed_batch.shape[0], -1).cpu().numpy().tolist()
                else:
                    # 使用CLIP编码
                    vectors = self.clip_vectors(batch)
                
                if vectors is None:
                    messagebox.showerror("Error", "Failed to compute sample image vectors")
                    return
                sample_vectors_list.extend(vectors)
            
            # 获取未知图像
            unknown_images = [info['image'] for info in self.molecule_info if not info.get('name')]
            
            # 处理未知图像
            unknown_vectors_list = []
            for i in range(0, len(unknown_images), max_parallel):
                batch = unknown_images[i:i + max_parallel]
                if encoding_method == "raw":
                    # 使用原始特征
                    processed_batch = [self.preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).to(self.device) 
                                    for img in batch]
                    processed_batch = torch.stack(processed_batch)
                    vectors = processed_batch.reshape(processed_batch.shape[0], -1).cpu().numpy().tolist()
                else:
                    # 使用CLIP编码
                    vectors = self.clip_vectors(batch)
                
                if vectors is None:
                    messagebox.showerror("Error", "Failed to compute unknown image vectors")
                    return
                unknown_vectors_list.extend(vectors)
            
            # 关联样本向量与名称和颜色
            sample_indices = [i for i, info in enumerate(self.molecule_info) if info.get('name')]
            for i, sample_index in enumerate(sample_indices):
                info = self.molecule_info[sample_index]
                for j in range(36):
                    sample_vectors[(info['name'], j)] = (sample_vectors_list[i * 36 + j], info['color'])
            
            # 如果使用自动阈值方法，计算阈值
            if threshold_method != "manual":
                # 计算所有相似度
                similarities = []
                for unknown_vector in unknown_vectors_list:
                    for (name, angle), (sample_vector, _) in sample_vectors.items():
                        similarity = self.calculate_similarity(unknown_vector, sample_vector)
                        similarities.append(similarity)
                similarities = np.array(similarities)
                
                # 根据选择的方法计算阈值
                if threshold_method == "otsu":
                    self.threshold = self.calculate_threshold_otsu(similarities)
                elif threshold_method == "gmm":
                    self.threshold = self.calculate_threshold_gmm(similarities)
                elif threshold_method == "distribution":
                    self.threshold = self.calculate_threshold_distribution(similarities)
                
                # 更新阈值显示
                self.threshold_entry.delete(0, tk.END)
                self.threshold_entry.insert(0, str(round(self.threshold, 3)))
            
            # 分类未知图像
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

            # 更新显示
            self.contour_image = self.contour_image.copy()
            for info in self.molecule_info:
                x, y, w, h = info['bbox']
                cv2.rectangle(self.contour_image, (x, y), (x+w, y+h), info.get('color', (255, 255, 255)), 1)

            pil_image = Image.fromarray(cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(pil_image)
            self.label.config(image=self.photo)

            # 显示分类结果
            result_message = "Classification Results:\n"
            for category, count in classified_count.items():
                result_message += f"{category}: {count}\n"
            result_message += f"\nEncoding Method: {encoding_method}\n"
            result_message += f"Threshold: {self.threshold:.3f}"

            result_dialog = Toplevel(self.root)
            result_dialog.title("Classification Results")
            result_dialog.geometry("400x300+500+350")

            tk.Label(result_dialog, text=result_message, wraplength=400).pack(pady=20)
            ok_button = tk.Button(result_dialog, text="OK", command=result_dialog.destroy)
            ok_button.pack()

        except Exception as e:
            messagebox.showerror("Error", f"Error during classification: {str(e)}")
            print(f"Error during classification: {str(e)}")
            return

def main():
    root = tk.Tk()
    root.title("Start")
    root.geometry("750x200+150+100")  # 主窗口位置
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
