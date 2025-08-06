import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from database import Database
import sqlite3
from PyQt5.QtWidgets import QApplication, QProgressDialog
from PyQt5.QtCore import Qt
#weqw
# Пути к папкам
result_dir = 'images_resized/result'
model_path = 'model/trained_model01.h5'
windows_dir = 'debug_windows'  # Новая папка для окон

# Создание директорий
os.makedirs(result_dir, exist_ok=True)
os.makedirs(windows_dir, exist_ok=True)

def save_window(window, x, y, prediction, index, debug_dir):
    """
    Сохраняет окно анализа для отладки и информацию для отчета
    """
    window_image = (window * 255).astype(np.uint8)
    filename = f"window_{index:04d}_x{x}_y{y}_pred{prediction:.3f}.png"
    filepath = os.path.join(debug_dir, filename)
    cv2.imwrite(filepath, window_image)
    return filename, x, y, prediction

def calculate_coordinates(center_lat, center_lon, center_alt, pixel_x, pixel_y, image_width, image_height):
    """
    Вычисляет координаты для точки на изображении
    Args:
        center_lat, center_lon: координаты центра изображения
        center_alt: высота центра
        pixel_x, pixel_y: координаты точки в пикселях
        image_width, image_height: размеры изображения в пикселях
    """
    # Получаем угол обзора камеры (в градусах)
    # Это нужно получать из метаданных камеры, пока используем примерные значения
    fov_horizontal = 60  # градусов
    fov_vertical = 40    # градусов
    
    # Вычисляем масштаб (градусов на пиксель)
    scale_lon = fov_horizontal / image_width
    scale_lat = fov_vertical / image_height
    
    # Вычисляем смещение от центра в пикселях
    dx = pixel_x - image_width/2
    dy = image_height/2 - pixel_y  # Инвертируем Y, так как в координатах север вверху
    
    # Вычисляем смещение в градусах
    dlon = dx * scale_lon
    dlat = dy * scale_lat
    
    # Корректируем масштаб долготы с учетом широты
    # На разных широтах градус долготы имеет разную длину
    lon_scale_correction = 1 / np.cos(np.radians(center_lat))
    dlon = dlon * lon_scale_correction
    
    # Вычисляем итоговые координаты
    lat = center_lat + dlat
    lon = center_lon + dlon
    alt = center_alt  # Высота остается той же, что и в центре
    
    print(f"Debug: pixel({pixel_x}, {pixel_y}) -> coord({lat:.6f}, {lon:.6f}, {alt:.1f})")
    print(f"Debug: scales(lat={scale_lat:.8f}°/px, lon={scale_lon:.8f}°/px)")
    
    return lat, lon, alt

def get_image_metadata(image_path):
    """Извлекает метаданные из изображения"""
    try:
        print(f"\nПопытка чтения метаданных из: {image_path}")
        
        metadata = {
            'creation_time': None,  # Изначально None
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0
        }
        
        # Пробуем разные способы получения даты
        with Image.open(image_path) as img:
            # 1. Пробуем получить EXIF
            date_from_exif = None
            try:
                exif = img._getexif()
                if exif:
                    print("Найдены EXIF данные")
                    
                    # Список тегов, где может быть дата
                    date_tags = [
                        36867,  # DateTimeOriginal
                        36868,  # DateTimeDigitized
                        306,    # DateTime
                        50971,  # DateTimeOriginal
                        37521   # SubsecDateTimeOriginal
                    ]
                    
                    # Перебираем все возможные теги с датой
                    for tag_id in date_tags:
                        if tag_id in exif:
                            date_str = exif[tag_id]
                            try:
                                # Пробуем разные форматы даты
                                if ':' in date_str:
                                    date_from_exif = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                                else:
                                    date_from_exif = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                                print(f"Найдена дата в EXIF (тег {tag_id}): {date_from_exif}")
                                break
                            except:
                                continue
                    
                    # Получаем GPS данные
                    if 'GPSInfo' in exif:
                        gps_data = {}
                        for gps_tag in exif['GPSInfo']:
                            sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                            gps_data[sub_tag] = exif['GPSInfo'][gps_tag]
                        
                        if all(k in gps_data for k in ['GPSLatitude', 'GPSLongitude']):
                            lat = convert_to_degrees(gps_data['GPSLatitude'])
                            lon = convert_to_degrees(gps_data['GPSLongitude'])
                            
                            if gps_data.get('GPSLatitudeRef') == 'S':
                                lat = -lat
                            if gps_data.get('GPSLongitudeRef') == 'W':
                                lon = -lon
                            
                            metadata['latitude'] = lat
                            metadata['longitude'] = lon
                            print(f"Найдены GPS координаты: lat={lat}, lon={lon}")
                        
                        if 'GPSAltitude' in gps_data:
                            alt = float(gps_data['GPSAltitude'].numerator) / float(gps_data['GPSAltitude'].denominator)
                            if gps_data.get('GPSAltitudeRef') == 1:
                                alt = -alt
                            metadata['altitude'] = alt
                            print(f"Найдена высота: {alt}")
            
            except Exception as e:
                print(f"Ошибка при чтении EXIF: {str(e)}")
            
            # 2. Пробуем получить дату из метаданных PNG
            if date_from_exif is None:
                try:
                    for key, value in img.info.items():
                        if any(date_key in key.lower() for date_key in ['date', 'time', 'creation']):
                            print(f"Найдены метаданные PNG: {key}: {value}")
                            try:
                                date_from_exif = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                                print(f"Получена дата из PNG метаданных: {date_from_exif}")
                                break
                            except:
                                continue
                except Exception as e:
                    print(f"Ошибка при чтении PNG метаданных: {str(e)}")
        
        if file_ext in ['.jpg', '.jpeg']:
            # Для JPEG используем EXIF
            with Image.open(image_path) as img:
                try:
                    exif = img._getexif()
                    if exif:
                        print("Найдены EXIF данные")
                        
                        for tag_id in exif:
                            tag = TAGS.get(tag_id, tag_id)
                            data = exif.get(tag_id)
                            
                            if tag == 'DateTimeOriginal':
                                try:
                                    metadata['creation_time'] = datetime.strptime(data, '%Y:%m:%d %H:%M:%S')
                                except:
                                    pass
                            
                            elif tag == 'GPSInfo':
                                gps_data = {}
                                for gps_tag in data:
                                    sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                                    gps_data[sub_tag] = data[gps_tag]
                                
                                if all(k in gps_data for k in ['GPSLatitude', 'GPSLongitude']):
                                    lat = convert_to_degrees(gps_data['GPSLatitude'])
                                    lon = convert_to_degrees(gps_data['GPSLongitude'])
                                    
                                    if gps_data.get('GPSLatitudeRef') == 'S':
                                        lat = -lat
                                    if gps_data.get('GPSLongitudeRef') == 'W':
                                        lon = -lon
                                    
                                    metadata['latitude'] = lat
                                    metadata['longitude'] = lon
                                
                                if 'GPSAltitude' in gps_data:
                                    alt = float(gps_data['GPSAltitude'].numerator) / float(gps_data['GPSAltitude'].denominator)
                                    if gps_data.get('GPSAltitudeRef') == 1:
                                        alt = -alt
                                    metadata['altitude'] = alt
                except Exception as e:
                    print(f"Ошибка при чтении EXIF: {str(e)}")
        
        elif file_ext == '.png':
            # Для PNG используем встроенные метаданные
            with Image.open(image_path) as img:
                try:
                    # Пробуем получить текстовые метаданные PNG
                    for key, value in img.info.items():
                        print(f"PNG metadata: {key}: {value}")
                        
                        if key.lower() in ['creation time', 'create_time']:
                            try:
                                metadata['creation_time'] = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                            except:
                                pass
                        
                        # Можно добавить другие ключи метаданных PNG
                except Exception as e:
                    print(f"Ошибка при чтении PNG метаданных: {str(e)}")
        
        # Пробуем получить время создания файла, если не удалось получить из метаданных
        if metadata['creation_time'] == datetime.now():
            try:
                creation_time = datetime.fromtimestamp(os.path.getctime(image_path))
                metadata['creation_time'] = creation_time
                print(f"Использовано время создания файла: {creation_time}")
            except Exception as e:
                print(f"Ошибка при получении времени создания файла: {str(e)}")
        
        print("\nИтоговые метаданные:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        return metadata
            
    except Exception as e:
        print(f"Ошибка при чтении метаданных: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'creation_time': datetime.now(),
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0
        }

def convert_to_degrees(value):
    """Конвертирует GPS координаты в градусы"""
    d = float(value[0].numerator) / float(value[0].denominator)
    m = float(value[1].numerator) / float(value[1].denominator)
    s = float(value[2].numerator) / float(value[2].denominator)
    return d + (m / 60.0) + (s / 3600.0)

def analyze_large_image(model, image_path, debug_dir, analysis_id=None, progress_callback=None, window_size=(224, 224), stride=168):
    """
    Анализирует большое изображение методом скользящего окна
    """
    # Создаем подпапку для текущего изображения
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_debug_dir = os.path.join(debug_dir, image_name)
    os.makedirs(image_debug_dir, exist_ok=True)
    
    # Очищаем только подпапку текущего изображения
    for file in os.listdir(image_debug_dir):
        os.remove(os.path.join(image_debug_dir, file))
    
    # Создаем новую запись анализа только если не передан analysis_id
    db = Database()
    if analysis_id is None:
        analysis_id = db.create_analysis()
    
    # Читаем метаданные изображения
    metadata = get_image_metadata(image_path)
    if metadata is None:
        print("Предупреждение: Не удалось прочитать метаданные изображения")
        center_lat, center_lon = 0, 0
        creation_time = None
        altitude = 0
    else:
        center_lat = metadata.get('latitude', 0)
        center_lon = metadata.get('longitude', 0)
        creation_time = metadata.get('creation_time', datetime.now())
        altitude = metadata.get('altitude', 0)
    
    # Добавляем информацию об изображении в базу
    image_id = db.add_image(
        file_path=image_path,
        date=creation_time,
        latitude=center_lat,
        longitude=center_lon,
        altitude=altitude,
        analysis_id=analysis_id
    )
    
    # После получения метаданных, выводим центральные координаты
    print(f"\nЦентральные координаты изображения:")
    print(f"Широта: {center_lat:.6f}")
    print(f"Долгота: {center_lon:.6f}\n")
    
    # Загрузка изображения
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Изменяем размер изображения
    scale_factor = 2.0
    new_width = int(original_image.shape[1] * scale_factor)
    new_height = int(original_image.shape[0] * scale_factor)
    original_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Создаем изображение для результатов
    result_image = original_image.copy()
    
    # Рисуем жирную точку в центре
    center_x = new_width // 2
    center_y = new_height // 2
    cv2.circle(
        result_image,
        (center_x, center_y),
        10,  # радиус точки
        (0, 0, 255),  # красный цвет (BGR)
        -1  # заполненный круг
    )
    
    # Добавляем центральные координаты на изображение
    center_text = f"Center: {center_lat:.6f}, {center_lon:.6f}"
    cv2.putText(
        result_image,
        center_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    height, width = original_image.shape[:2]
    detection_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Нормализация изображения
    normalized_image = original_image.astype(np.float32) / 255.0
    
    # Создаем список для отчета
    windows_report = []
    
    print("Анализ изображения...")
    print(f"Размер изображения: {width}x{height}")
    print(f"Размер окна: {window_size[0]}x{window_size[1]}")
    print(f"Шаг: {stride}")
    print(f"Окна сохраняются в папку: {image_debug_dir}")
    
    # Правильный расчет количества окон
    num_windows_y = (height - window_size[0]) // stride + 1
    num_windows_x = (width - window_size[1]) // stride + 1
    total_windows = num_windows_x * num_windows_y
    
    print(f"Количество окон по X: {num_windows_x}")
    print(f"Количество окон по Y: {num_windows_y}")
    print(f"Всего окон: {total_windows}")
    
    processed_windows = 0
    detected_count = 0
    
    # Скользящее окно
    for y in range(0, height - window_size[0] + 1, stride):
        for x in range(0, width - window_size[1] + 1, stride):
            window = normalized_image[y:y + window_size[0], x:x + window_size[1]]
            
            window_batch = np.expand_dims(window, axis=0)
            prediction = model.predict(window_batch, verbose=0)[0][0]
            
            processed_windows += 1
            
            # Обновляем прогресс каждое окно
            if processed_windows % 2 == 0 or processed_windows == total_windows:
                print(f"Прогресс: {processed_windows}/{total_windows} | Найдено: {detected_count}")
                if progress_callback:
                    status_text = f"Текущее изображение: {os.path.basename(image_path)}"
                    progress_callback(status_text, processed_windows, total_windows)
            
            if prediction > 0.5:
                # Вычисляем координаты центра окна
                center_x = x + window_size[0]/2
                center_y = y + window_size[1]/2
                
                # Получаем географические координаты
                lat, lon, alt = calculate_coordinates(
                    center_lat, 
                    center_lon,
                    center_y,
                    center_x,
                    center_y,
                    width,
                    height
                )
                
                # Рисуем прямоугольник
                cv2.rectangle(
                    result_image,
                    (x, y),
                    (x + window_size[0], y + window_size[1]),
                    (0, 255, 0),
                    2
                )
                
                # Добавляем информацию в столбик с увеличенным шрифтом
                font_size = 1.0  # Увеличенный размер шрифта
                line_height = 30  # Расстояние между строками
                
                # Уверенность
                cv2.putText(
                    result_image,
                    f"Conf: {prediction:.2f}",
                    (x, y-line_height*2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (0, 255, 0),
                    2
                )
                
                # Широта
                cv2.putText(
                    result_image,
                    f"Lat: {lat:.6f}",
                    (x, y-line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (0, 255, 0),
                    2
                )
                
                # Долгота
                cv2.putText(
                    result_image,
                    f"Lon: {lon:.6f}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (0, 255, 0),
                    2
                )
                
                detected_count += 1
                
                # Сохраняем окно для отчета в подпапку изображения
                filename, _, _, _ = save_window(window, x, y, prediction, processed_windows, image_debug_dir)
                windows_report.append({
                    'filename': filename,
                    'x': x,
                    'y': y,
                    'prediction': prediction,
                    'window_index': processed_windows,
                    'coordinates': {
                        'latitude': lat,
                        'longitude': lon
                    }
                })
    
    print(f"Всего найдено областей: {detected_count}")
    
    # Вместо создания CSV отчета сохраняем колонии в базу
    for item in windows_report:
        # Получаем путь к окну
        window_path = os.path.join(image_debug_dir, item['filename'])
        
        # Сохраняем в базу данных с координатами на изображении
        db.add_colony(
            image_id=image_id,
            latitude=item['coordinates']['latitude'],
            longitude=item['coordinates']['longitude'],
            confidence=float(item['prediction']),
            x=int(item['x'] / scale_factor),  # Делим на scale_factor для получения оригинальных координат
            y=int(item['y'] / scale_factor),
            window_path=window_path,
            verified=False
        )
    
    # Возвращаем изображение в исходный размер
    result_image = cv2.resize(result_image, (int(width/scale_factor), int(height/scale_factor)))
    
    # В конце анализа отправляем сигнал о завершении изображения
    if progress_callback:
        status_text = (f"Завершена обработка: {os.path.basename(image_path)}\n"
                      f"Обработано окон: {total_windows}\n"
                      f"Найдено областей: {detected_count}")
        progress_callback(status_text, total_windows, total_windows)
    
    return result_image, detection_mask

def process_single_image(model, image_path):
    """
    Обрабатывает одно изображение и сохраняет результат
    """
    try:
        result_image, detection_mask = analyze_large_image(model, image_path, windows_dir)
        
        base_name = os.path.basename(image_path)
        result_path = os.path.join(result_dir, f'processed_{base_name}')
        mask_path = os.path.join(result_dir, f'mask_{base_name}')
        
        cv2.imwrite(result_path, result_image)
        cv2.imwrite(mask_path, detection_mask)
        
        print(f"Результат сохранен в {result_path}")
        print(f"Маска сохранена в {mask_path}")
        
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {str(e)}")

def analyze_new_images(model, image_folder, progress_callback=None):
    """
    Анализирует все изображения в указанной папке
    """
    # Загружаем модель здесь, если она не передана
    if model is None:
        if not os.path.exists(model_path):
            raise ValueError(f"Модель не найдена по пути {model_path}")
        print("Загрузка модели...")
        model = load_model(model_path)
        print("Модель загружена успешно")

    # Создаем папки для результатов внутри папки анализа
    result_dir = os.path.join(image_folder, 'results')
    debug_dir = os.path.join(image_folder, 'debug_windows')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Получаем список всех изображений
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_folder, filename)
        print(f"Обработка {filename}... ({i+1}/{len(image_files)})")
        
        try:
            result_image, detection_mask = analyze_large_image(
                model, 
                image_path, 
                debug_dir,
                progress_callback
            )
            
            # Сохраняем результаты
            result_path = os.path.join(result_dir, f'processed_{filename}')
            mask_path = os.path.join(result_dir, f'mask_{filename}')
            
            cv2.imwrite(result_path, result_image)
            cv2.imwrite(mask_path, detection_mask)
            
            print(f"Результат сохранен в {result_path}")
            print(f"Маска сохранена в {mask_path}")
            
            # Сигнализируем о завершении обработки изображения
            if progress_callback:
                progress_callback("NEXT_IMAGE", None, None)
            
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {str(e)}")

def analyze_images(image_paths, debug_dir):
    """Анализирует группу изображений в рамках одного анализа"""
    try:
        # Создаем прогресс-диалог сразу
        progress = QProgressDialog("Подготовка к анализу...", "Отмена", 0, 100)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()  # Явно показываем диалог
        QApplication.processEvents()  # Обновляем UI
        
        # Загружаем модель
        progress.setLabelText("Загрузка модели...")
        if not os.path.exists(model_path):
            raise ValueError(f"Модель не найдена по пути {model_path}")
        model = load_model(model_path)
        
        # Создаем новую запись анализа
        db = Database()
        analysis_id = db.create_analysis()
        print(f"Создан новый анализ с ID: {analysis_id}")
        
        # Создаем папку для окон отладки
        os.makedirs(debug_dir, exist_ok=True)
        
        # Устанавливаем максимум для общего прогресса
        total_progress = len(image_paths) * 100  # 100% на каждое изображение
        progress.setMaximum(total_progress)
        
        for i, image_path in enumerate(image_paths):
            if progress.wasCanceled():
                break
            
            base_progress = i * 100  # Базовый прогресс для текущего изображения
            
            # Обновляем прогресс-бар для текущего изображения
            def update_progress(text, current=None, total=None):
                if current is not None and total is not None:
                    # Вычисляем процент выполнения текущего изображения
                    image_progress = (current / total) * 100
                    # Добавляем к базовому прогрессу
                    total_value = int(base_progress + image_progress)
                    progress.setValue(total_value)
                
                progress.setLabelText(
                    f"Обработка изображения {i+1} из {len(image_paths)}:\n"
                    f"{os.path.basename(image_path)}\n{text}"
                )
                QApplication.processEvents()
            
            # Анализируем изображение
            result_image, detection_mask = analyze_large_image(
                model=model,
                image_path=image_path,
                debug_dir=debug_dir,
                analysis_id=analysis_id,
                progress_callback=update_progress
            )
            
            # Сохраняем результат
            result_dir = os.path.join(os.path.dirname(image_path), 'results')
            os.makedirs(result_dir, exist_ok=True)
            
            base_name = os.path.basename(image_path)
            result_path = os.path.join(result_dir, f'processed_{base_name}')
            cv2.imwrite(result_path, result_image)
            
            print(f"Результат сохранен: {result_path}")
        
        progress.setValue(total_progress)  # Устанавливаем 100% прогресс
        progress.close()
        print(f"Анализ завершен успешно. ID анализа: {analysis_id}")
        return analysis_id
        
    except Exception as e:
        if 'progress' in locals():
            progress.close()
        print(f"Ошибка при анализе изображений: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Проверяем наличие обученной модели
    if not os.path.exists(model_path):
        print(f"Ошибка: Модель не найдена по пути {model_path}")
        print("Сначала запустите train_model.py для обучения модели")
        exit(1)
    
    # Загружаем обученную модель
    print("Загрузка модели...")
    model = load_model(model_path)
    print("Модель загружена успешно")
    
    # Анализ конкретного изображения
    target_image = "images/3_with_metadata.png"
    if os.path.exists(target_image):
        print(f"Анализ изображения {target_image}...")
        process_single_image(model, target_image)
    else:
        print(f"Файл {target_image} не найден!")
    
    print("Обработка завершена!")  