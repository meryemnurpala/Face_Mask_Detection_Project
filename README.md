# 😷 Face Mask Detection - Transfer Learning Project

Bu proje, **Transfer Learning** kullanarak insan yüzlerinde maske tespiti yapan bir **image classification** modeli geliştirmektedir. **MobileNetV2** tabanlı model, **pre-trained ImageNet weights** kullanarak **masked/unmasked** yüzleri sınıflandırır.

## 📋 Proje Özeti

### 🎯 Amaç
- Transfer Learning ile maske tespiti modeli geliştirmek
- Yüksek doğruluk oranında sınıflandırma yapmak
- Kullanıcı dostu arayüz ile test imkanı sağlamak

### 🔬 Kullanılan Teknolojiler
- **Model**: MobileNetV2 (Transfer Learning)
- **Framework**: TensorFlow/Keras
- **Dataset**: Face Mask Detection Dataset
- **GUI**: Tkinter (Python Interface)
- **Image Processing**: PIL (Pillow)

## 📁 Proje Yapısı

```
Project/
├── data/
│   ├── with_mask/          # Masked face images
│   └── without_mask/       # Unmasked face images
├── face_mask_interface/
│   └── app.py             # Tkinter GUI application
├── face_mask_detection.py # Main model training script
├── requirements.txt       # Required libraries
├── README.md             # This file
└── Saved Files:
    ├── face_mask_model.keras    # Trained model
    ├── processed_data.pkl       # Processed dataset
    ├── training_history.pkl     # Training history
    ├── training_curves.png      # Training curves
    ├── confusion_matrix.png     # Confusion matrix
    ├── classification_report.txt # Classification report
    └── sample_predictions.png   # Sample predictions
```

## 🚀 Kurulum

### 1. Gereksinimler
- Python 3.10
- Anaconda (önerilen) veya pip

### 2. Ortam Kurulumu
```bash
# Anaconda ile yeni ortam oluştur
conda create -n face_mask python=3.10
conda activate face_mask

# Gerekli kütüphaneleri yükle
pip install -r requirements.txt
```

### 3. Dataset
Proje, Face Mask Detection Dataset kullanmaktadır:
- **Source**: [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset?resource=download)
- **Classes**: With Mask (😷) / Without Mask (😮)
- **Location**: `data/with_mask/` ve `data/without_mask/` klasörleri

## 📊 Model Eğitimi

### 1. Model Eğitimi Çalıştırma
```bash
python face_mask_detection.py
```

### 2. Training Process
- **Data Preprocessing**: Images resized to 224x224 and normalized
- **Data Split**: 80% training, 20% validation
- **Model Architecture**: MobileNetV2 + Dense Layers
- **Training Parameters**:
  - Loss: binary_crossentropy
  - Optimizer: Adam (lr=1e-4)
  - Epochs: 15 (with Early Stopping)
  - Batch Size: 32

### 3. Model Saving
- Trained model saved as `face_mask_model.keras`
- Processed data saved as `processed_data.pkl`
- Training history saved as `training_history.pkl`
- 
![Ekran görüntüsü 2025-07-05 001051](https://github.com/user-attachments/assets/d5f618cf-466c-48e9-b16e-1d39038cdd25)

![Ekran görüntüsü 2025-07-05 000416](https://github.com/user-attachments/assets/e8fac273-6df0-4c00-b21c-10024845fdf6)



## 🖥️ Arayüz Kullanımı

### 1. Arayüzü Başlatma
```bash
cd face_mask_interface
python app.py
```

### 2. Kullanım Adımları
1. **Fotoğraf Seç**: "📸 Fotoğraf Seç" butonuna tıklayın
2. **Görsel Yükle**: Yüz içeren bir fotoğraf seçin
3. **Analiz Et**: "🔍 Analiz Et" butonuna tıklayın
4. **Sonucu Gör**: Maske durumu ve doğruluk oranını inceleyin

   ![Ekran görüntüsü 2025-07-05 013106](https://github.com/user-attachments/assets/c4cf7f58-a67d-489a-b448-4a05ec5c703d)


### 3. Interface Features
- ✅ **Flexible Size Support**: Any image size
- ✅ **Aspect Ratio Preservation**: Proportions maintained
- ✅ **Image Preview**: Display uploaded image
- ✅ **Confidence Score**: Accuracy percentage
- ✅ **Progress Bar**: Visual confidence display
- ✅ **Model Information**: Technical details

## 📈 Model Performance

### Training Metrics
- **Accuracy**: ~95% (on validation set)
- **Loss**: Low overfitting
- **Confusion Matrix**: High precision/recall

- ![Ekran görüntüsü 2025-07-05 013144](https://github.com/user-attachments/assets/c8430803-91bd-45a1-afbf-6498d06bcfe4)

  ![Ekran görüntüsü 2025-07-05 013152](https://github.com/user-attachments/assets/61432801-b518-47ab-8dc4-9170002a8bc4)

![Ekran görüntüsü 2025-07-05 013136](https://github.com/user-attachments/assets/9931eac7-942a-4e73-8797-7e7dfe5dfea5)


### Classification Results
- **With Mask**: High accuracy
- **Without Mask**: High accuracy
- **F1-Score**: Balanced performance

## 🔧 Teknik Detaylar

### Model Architecture
```
MobileNetV2 (pretrained)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='sigmoid')
```

### Transfer Learning Strategy
- **Base Model**: MobileNetV2 (ImageNet weights)
- **Freezing**: Base model frozen
- **Fine-tuning**: Only top layers trained

### Data Preprocessing
- **Resizing**: 224x224 pixels
- **Normalization**: 0-1 range
- **Augmentation**: Applied during training

## 📋 Ödev Gereksinimleri Karşılanması

### ✅ Tamamlanan Maddeler
1. ✅ Verileri %80 eğitim, %20 doğrulama olarak ayırma
2. ✅ Görselleri 224x224 boyutuna getirme ve normalize etme
3. ✅ Transfer learning ile model eğitimi (MobileNetV2)
4. ✅ Pretrained ağı freeze etme ve dense layer ekleme
5. ✅ Binary crossentropy loss fonksiyonu
6. ✅ Accuracy metric kullanımı
7. ✅ Epoch-loss ve accuracy grafikleri
8. ✅ Confusion matrix oluşturma
9. ✅ Precision, recall, f1-score hesaplama
10. ✅ Model tahminlerini örnek görseller üzerinde gösterme

### 📊 Report Content
- **Model Architecture**: MobileNetV2 + Dense Layers
- **Training Process**: Transfer Learning strategy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualizations**: Training curves, confusion matrix, sample predictions
- **Challenges**: Overfitting prevention, data imbalance solution

## 🐛 Sorun Giderme

### Yaygın Hatalar
1. **Model Bulunamadı**: Önce `face_mask_detection.py` çalıştırın
2. **Kütüphane Hatası**: `pip install -r requirements.txt` çalıştırın
3. **NumPy Uyumsuzluğu**: `pip install 'numpy<2'` kullanın

### Performance Optimization
- **GPU Usage**: CUDA-enabled TensorFlow installation
- **Batch Size**: Adjust according to your system
- **Epoch Count**: Optimize with early stopping

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun


## 👨‍💻 Geliştirici

**Öğrenci**: Meryemnur PALA,Ezgi Kutlu, Nisan Demiray
**Ders**: Görüntü İşleme
**Konu**: Transfer Learning ile Maske Tespiti

---

**Not**: Bu proje, Turkcell kapsamında transfer learning kavramlarını öğrenmek ve pratik yapmak amacıyla geliştirilmiştir. Gerçek dünya uygulamaları için ek optimizasyonlar gerekebilir. 
