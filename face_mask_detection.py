import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
import pickle

# Model ve veri dosyalarının yolları
model_path = 'face_mask_model.keras'
data_path = 'processed_data.pkl'

# Eğer kaydedilmiş model ve veri varsa yükle, yoksa yeni oluştur
if os.path.exists(model_path) and os.path.exists(data_path):
    print('Kaydedilmiş model ve veri bulundu. Yükleniyor...')
    
    # Modeli yükle
    model = load_model(model_path)
    print('Model başarıyla yüklendi!')
    
    # Veriyi yükle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']
    
    print(f'Veri yüklendi: Eğitim seti: {len(X_train)}, Doğrulama seti: {len(X_val)}')
    
    # Eğitim geçmişini yükle (grafikler için)
    if os.path.exists('training_history.pkl'):
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        print('Eğitim geçmişi yüklendi.')
    else:
        print('Eğitim geçmişi bulunamadı. Grafikler oluşturulamayacak.')
        history = None
    
else:
    print('Kaydedilmiş model bulunamadı. Yeni model oluşturuluyor...')
    
    # 1. Veri Hazırlama
    with_mask_dir = 'data/with_mask'
    without_mask_dir = 'data/without_mask'
    image_size = (224, 224)
    images = []
    labels = []

    print('Görseller yükleniyor...')
    # With mask
    for img_name in os.listdir(with_mask_dir):
        img_path = os.path.join(with_mask_dir, img_name)
        try:
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(1)
        except Exception as e:
            print(f'Hata (with_mask): {img_path} - {e}')
    # Without mask
    for img_name in os.listdir(without_mask_dir):
        img_path = os.path.join(without_mask_dir, img_name)
        try:
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(0)
        except Exception as e:
            print(f'Hata (without_mask): {img_path} - {e}')

    images = np.array(images)
    labels = np.array(labels)
    print(f'Toplam görsel: {len(images)}')
    print(f'With mask: {np.sum(labels==1)}, Without mask: {np.sum(labels==0)}')

    # Eğitim ve doğrulama setlerine ayırma
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    print(f'Eğitim seti: {len(X_train)}, Doğrulama seti: {len(X_val)}')
    
    # Veriyi kaydet
    with open(data_path, 'wb') as f:
        pickle.dump({'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val}, f)
    print('Veri kaydedildi.')

    # 2. Model Kurulumu (MobileNetV2 + Dense Layer)
    print('Model oluşturuluyor...')
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False  # Freeze
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # 3. Model Eğitimi
    print('Model eğitiliyor...')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history_obj = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=[early_stop]
    )
    
    # Modeli kaydet
    model.save(model_path)
    print(f'Model {model_path} olarak kaydedildi.')
    
    # Eğitim geçmişini kaydet
    history = history_obj.history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print('Eğitim geçmişi kaydedildi.')

# 4. Eğitim ve Doğrulama Sonuçları
if history is not None:
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['loss'], label='Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    print('Eğitim grafikleri training_curves.png olarak kaydedildi.')
else:
    print('Eğitim geçmişi bulunamadığı için grafikler oluşturulamadı.')

# 5. Değerlendirme
print('Model doğrulama setinde değerlendiriliyor...')
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_val, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Without Mask', 'With Mask'])
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
print('Confusion matrix confusion_matrix.png olarak kaydedildi.')

print('Classification Report:')
report = classification_report(y_val, y_pred, target_names=['Without Mask', 'With Mask'], digits=4)
print(report)
with open('classification_report.txt', 'w') as f:
    f.write(report)

# 6. Örnek Tahminler
print('Örnek tahminler kaydediliyor...')
example_indices = random.sample(range(len(X_val)), 5)
plt.figure(figsize=(15,5))
for i, idx in enumerate(example_indices):
    img = X_val[idx]
    true_label = 'With Mask' if y_val[idx]==1 else 'Without Mask'
    pred_label = 'With Mask' if y_pred[idx]==1 else 'Without Mask'
    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'True: {true_label}\nPred: {pred_label}')
plt.tight_layout()
plt.savefig('sample_predictions.png')
plt.close()
print('5 örnek tahmin sample_predictions.png olarak kaydedildi.')

print('\nTüm sonuçlar başarıyla kaydedildi! Rapor için:\n- training_curves.png\n- confusion_matrix.png\n- classification_report.txt\n- sample_predictions.png\ndosyalarını kullanabilirsin.')

model.save("model.h5")