import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import sys

# Ana dizindeki modeli kullanabilmek için path ekle
sys.path.append('..')

class FaceMaskDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Detection")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Model yükle
        self.model = self.load_model()
        if self.model is None:
            messagebox.showerror("Hata", "Model yüklenemedi!")
            root.destroy()
            return
        
        # Değişkenler
        self.image_path = None
        self.photo = None
        
        self.create_widgets()
    
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            model_path = '../face_mask_model.keras'
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                return model
            else:
                messagebox.showerror("Hata", "Model dosyası bulunamadı!")
                return None
        except Exception as e:
            messagebox.showerror("Hata", f"Model yüklenirken hata: {e}")
            return None
    
    def create_widgets(self):
        # Ana başlık
        title_label = tk.Label(
            self.root, 
            text="😷 Face Mask Detection", 
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Ana frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Sol panel (fotoğraf seçimi)
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Fotoğraf seç butonu
        select_btn = tk.Button(
            left_frame,
            text="📸 Fotoğraf Seç",
            command=self.select_image,
            font=("Arial", 12),
            bg='#3498db',
            fg='white',
            relief='flat',
            padx=20,
            pady=10
        )
        select_btn.pack(pady=10)
        
        # Fotoğraf gösterme alanı
        self.image_label = tk.Label(
            left_frame,
            text="Fotoğraf seçin...",
            bg='white',
            relief='solid',
            width=50,
            height=25
        )
        self.image_label.pack(pady=10)
        
        # Sağ panel (sonuçlar)
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Analiz butonu
        self.analyze_btn = tk.Button(
            right_frame,
            text="🔍 Analiz Et",
            command=self.analyze_image,
            font=("Arial", 12),
            bg='#27ae60',
            fg='white',
            relief='flat',
            padx=20,
            pady=10,
            state='disabled'
        )
        self.analyze_btn.pack(pady=10)
        
        # Sonuç alanı
        result_frame = tk.Frame(right_frame, bg='white', relief='solid', bd=2)
        result_frame.pack(fill='both', expand=True, pady=10)
        
        # Sonuç başlığı
        self.result_title = tk.Label(
            result_frame,
            text="Sonuç",
            font=("Arial", 16, "bold"),
            bg='white',
            fg='#2c3e50'
        )
        self.result_title.pack(pady=10)
        
        # Sonuç metni
        self.result_text = tk.Label(
            result_frame,
            text="Fotoğraf seçin ve analiz edin",
            font=("Arial", 12),
            bg='white',
            fg='#7f8c8d',
            wraplength=300
        )
        self.result_text.pack(pady=10)
        
        # Doğruluk oranı
        self.confidence_label = tk.Label(
            result_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg='white',
            fg='#2c3e50'
        )
        self.confidence_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            result_frame,
            orient='horizontal',
            length=300,
            mode='determinate'
        )
        self.progress.pack(pady=10)
        
        # Model bilgileri
        info_frame = tk.Frame(result_frame, bg='#ecf0f1', relief='solid', bd=1)
        info_frame.pack(fill='x', padx=10, pady=10)
        
        info_text = """
Model Bilgileri:
• Model: MobileNetV2
• Transfer Learning: Evet
• Veri Seti: Face Mask Detection
• Sınıflar: With Mask / Without Mask
        """
        
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=("Arial", 10),
            bg='#ecf0f1',
            fg='#2c3e50',
            justify='left'
        )
        info_label.pack(pady=10)
    
    def select_image(self):
        """Fotoğraf seç"""
        file_path = filedialog.askopenfilename(
            title="Fotoğraf Seç",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state='normal')
            self.clear_results()
    
    def display_image(self, file_path):
        """Fotoğrafı göster"""
        try:
            # Fotoğrafı yükle
            image = Image.open(file_path)
            
            # Orijinal boyutları al
            original_width, original_height = image.size
            
            # Maksimum görüntüleme boyutu (arayüz için)
            max_display_size = (400, 400)
            
            # Görüntüleme için boyutlandır (aspect ratio korunarak)
            display_image = image.copy()
            display_image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
            
            # Tkinter için dönüştür
            self.photo = ImageTk.PhotoImage(display_image)
            
            # Label'a yerleştir
            self.image_label.config(image=self.photo, text="")
            
            # Boyut bilgisini göster
            size_info = f"Orijinal: {original_width}x{original_height}"
            self.image_label.config(text=size_info)
            
        except Exception as e:
            messagebox.showerror("Hata", f"Fotoğraf yüklenirken hata: {e}")
    
    def analyze_image(self):
        """Fotoğrafı analiz et"""
        if not self.image_path:
            return
        
        try:
            # Fotoğrafı yükle ve işle
            image = Image.open(self.image_path)
            processed_image = self.preprocess_image(image)
            
            # Tahmin yap
            confidence = self.predict_mask(processed_image)
            
            # Sonucu göster
            self.show_results(confidence)
            
        except Exception as e:
            messagebox.showerror("Hata", f"Analiz sırasında hata: {e}")
    
    def preprocess_image(self, image):
        """Görseli model için hazırla"""
        try:
            # Orijinal boyutları al
            original_width, original_height = image.size
            
            # Model için gerekli boyut
            target_size = (224, 224)
            
            # Aspect ratio'yu koruyarak resize et
            # Önce en büyük boyutu hedef boyuta getir
            ratio = min(target_size[0] / original_width, target_size[1] / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Resize et
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 224x224 boyutunda beyaz arka plan oluştur
            final_image = Image.new('RGB', target_size, (255, 255, 255))
            
            # Resize edilmiş görseli ortala
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            final_image.paste(resized_image, (paste_x, paste_y))
            
            # Numpy array'e çevir ve normalize et
            image_array = img_to_array(final_image) / 255.0
            # Batch dimension ekle
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            # Hata durumunda basit resize
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
    
    def predict_mask(self, image_array):
        """Maske tespiti yap"""
        prediction = self.model.predict(image_array, verbose=0)
        probability = prediction[0][0]
        return probability
    
    def show_results(self, confidence):
        """Sonuçları göster"""
        # Sonucu belirle
        if confidence > 0.5:
            result = "😷 With Mask (Maskeli)"
            result_color = '#27ae60'
            emoji = "🟢"
        else:
            result = "😮 Without Mask (Maskesiz)"
            result_color = '#e74c3c'
            emoji = "🔴"
            confidence = 1 - confidence  # Doğru sınıf için confidence
        
        # Sonuçları güncelle
        self.result_title.config(text=f"{emoji} Sonuç")
        self.result_text.config(text=result, fg=result_color)
        
        # Doğruluk oranı
        confidence_percent = confidence * 100
        self.confidence_label.config(
            text=f"Doğruluk Oranı: {confidence_percent:.1f}%",
            fg=result_color
        )
        
        # Progress bar
        self.progress['value'] = confidence_percent
        self.progress['maximum'] = 100
    
    def clear_results(self):
        """Sonuçları temizle"""
        self.result_title.config(text="Sonuç")
        self.result_text.config(text="Fotoğraf seçin ve analiz edin", fg='#7f8c8d')
        self.confidence_label.config(text="")
        self.progress['value'] = 0

def main():
    root = tk.Tk()
    app = FaceMaskDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main() 