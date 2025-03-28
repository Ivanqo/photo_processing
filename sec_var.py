import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
from tqdm import tqdm
import cv2

# Инициализация детектора лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_brightness(img):
    """Определяет яркость области лица"""
    try:
        gray = np.array(img.convert('L'))
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_region = gray[y:y+h, x:x+w]
            return np.median(face_region[face_region > 0])
    except:
        pass
    return None

def calibrate_thresholds(image_folder, sample_size=20):
    """Калибрует пороги по яркости лиц"""
    brightness_values = []
    for filename in os.listdir(image_folder)[:sample_size]:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(os.path.join(image_folder, filename))
                brightness = detect_face_brightness(img) or np.median(np.array(img.convert('L')))
                brightness_values.append(brightness)
            except:
                continue
    
    if not brightness_values:
        return {'dark': 60, 'bright': 180}
    
    brightness_values = np.array(brightness_values)
    return {
        'dark': np.percentile(brightness_values, 25),
        'bright': np.percentile(brightness_values, 75)
    }

def classify_image(img, thresholds):
    """Классифицирует изображение по яркости лица"""
    face_brightness = detect_face_brightness(img)
    if face_brightness is None:
        gray = np.array(img.convert('L'))
        face_brightness = np.median(gray[gray > 0]) if np.any(gray > 0) else 0
    
    if face_brightness < thresholds['dark']:
        return "dark"
    elif face_brightness > thresholds['bright']:
        return "overexposed"
    return "normal"

def enhance_quality(img):
    """Улучшает качество с учетом типа изображения"""
    img_type = classify_image(img, THRESHOLDS)
    enhanced = img.copy()
    
    if img_type == "dark":
        # Осветление через LAB-пространство
        lab = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab = cv2.merge((clahe.apply(l), a, b))
        enhanced = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    elif img_type == "overexposed":
        # Легкое затемнение
        enhanced = ImageEnhance.Brightness(enhanced).enhance(0.9)
    
    # Умеренная резкость для всех
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    return enhanced

def process_images(input_folder, output_folder):
    """Обрабатывает и сохраняет изображения в PNG"""
    global THRESHOLDS
    THRESHOLDS = calibrate_thresholds(input_folder)
    print(f"Пороги яркости: Dark < {THRESHOLDS['dark']:.1f}, Bright > {THRESHOLDS['bright']:.1f}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        try:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"enhanced_{os.path.splitext(filename)[0]}.png")
            
            with Image.open(input_path) as img:
                # Удаление фона и конвертация в RGBA
                no_bg = remove(img.convert('RGBA'))
                
                # Улучшение качества
                enhanced = enhance_quality(no_bg.convert('RGB'))
                enhanced_rgba = Image.merge('RGBA', (*enhanced.split(), no_bg.getchannel('A')))
                
                # Сохранение в PNG с прозрачностью
                enhanced_rgba.save(output_path, format='PNG', compress_level=9)
                
        except Exception as e:
            print(f"Ошибка обработки {filename}: {str(e)}")

# Настройки

# Настройки
INPUT_FOLDER = r"C:\Users\Ivan\Downloads\photo"
OUTPUT_FOLDER = r"C:\Users\Ivan\Downloads\end"
THRESHOLDS = None  # Инициализируется в process_images

if __name__ == "__main__":
    process_images(INPUT_FOLDER, OUTPUT_FOLDER)