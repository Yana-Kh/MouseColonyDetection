import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, metrics
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5 import QtWidgets, QtGui, QtCore
import tensorflow as tf

# Пути к папкам
availability_dir = 'images_resized\\availability'
absence_dir = 'images_resized\\absence'
result_dir = 'images_resized\\result'
report_dir = 'report'

# Создание директории result и report, если их нет
os.makedirs(result_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)


# Подготовка данных
def load_images_from_folder(folder, label):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            filepath = os.path.join(folder, filename)
            img = load_img(filepath, target_size=(224, 224))  # Меняем размер на 224x224
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
            filenames.append(filename)
    return np.array(images), np.array(labels), filenames


# Загрузка изображений и меток
availability_images, availability_labels, availability_files = load_images_from_folder(availability_dir, 1)
absence_images, absence_labels, absence_files = load_images_from_folder(absence_dir, 0)

# Объединение данных
images = np.concatenate([availability_images, absence_images], axis=0)
labels = np.concatenate([availability_labels, absence_labels], axis=0)
filenames = availability_files + absence_files

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp, filenames_train, filenames_temp = train_test_split(
    images, labels, filenames, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test, filenames_val, filenames_test = train_test_split(
    X_temp, y_temp, filenames_temp, test_size=0.5, random_state=42
)

# Аугментация данных
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

# Построение улучшенной модели
def build_model():
    """
    Создает упрощенную модель на основе MobileNetV2
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)  # Уменьшаем размер входа
    )
    
    # Замораживаем базовую модель
    base_model.trainable = True
    fine_tune_at = 20  # Примерное количество слоев от конца
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )
    
    return model

# Создание и обучение модели
model = build_model()

# Обучение с колбэками
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# Генераторы данных
train_gen = train_datagen.flow(X_train, y_train, batch_size=16)
val_gen = val_datagen.flow(X_val, y_val, batch_size=16)

# Обучение модели
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    verbose=1  # Добавляем вывод прогресса
)

# Оценка точности на тестовой выборке
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
print(f'Точность на тестовой выборке: {test_accuracy:.4f}')
print(f'Точность: {test_precision:.4f}, Отзыв: {test_recall:.4f}')

# Предсказания на тестовом наборе
predictions = model.predict(X_test)
threshold = 0.5  # Порог для классификации
predicted_labels = (predictions > threshold).astype(int)

# Генерация отчёта в CSV
report_data = {
    'Filename': filenames_test,
    'True Label': y_test,
    'Predicted Label': predicted_labels.flatten(),
    'Prediction Confidence': predictions.flatten()
}
report_df = pd.DataFrame(report_data)
report_csv_path = os.path.join(report_dir, 'detection_report.csv')
report_df.to_csv(report_csv_path, index=False)
print(f"Отчёт сохранён в {report_csv_path}")

# # Визуализация и сохранение изображений с обнаруженными колониями
# for i, (filename, true_label, predicted_label, confidence) in enumerate(
#         zip(filenames_test, y_test, predicted_labels, predictions)
# ):
#     if predicted_label == 1:  # Если обнаружена колония
#         try:
#             img_path = os.path.join(availability_dir if true_label == 1 else absence_dir, filename)
#             if not os.path.exists(img_path):
#                 print(f"Файл не найден: {img_path}")
#                 continue
                
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"Не удалось загрузить изображение: {img_path}")
#                 continue

#             # Преобразуем изображение в HSV для поиска коричневого цвета
#             hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             lower_brown = np.array([10, 50, 50])
#             upper_brown = np.array([30, 255, 255])
#             mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

#             # Применяем маску к изображению
#             result = cv2.bitwise_and(img, img, mask=mask)

        # # Конвертируем в серый масштаб для нахождения контуров
        # gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if contours:
        #     # Выбираем самую большую область
        #     largest_contour = max(contours, key=cv2.contourArea)
        #     cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)

#             # Добавляем информацию о предсказании на изображение
#             cv2.putText(
#                 img,
#                 f'Confidence: {confidence[0]:.2f}',
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 2
#             )

        # # Сохраняем изображение с контуром
        # result_path = os.path.join(result_dir, filename)
        # cv2.imwrite(result_path, img)

#         except Exception as e:
#             print(f"Ошибка при обработке {filename}: {str(e)}")
#             continue

print("Готово! Изображения с обнаруженными колониями обработаны и сохранены.")

def analyze_large_image(model, image_path, window_size=(112, 112), stride=56):
    """
    Анализирует большое изображение методом скользящего окна
    """
    # Загрузка изображения
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Изменяем размер изображения
    scale_factor = 2.0  # Увеличиваем изображение в 2 раза
    new_width = int(original_image.shape[1] * scale_factor)
    new_height = int(original_image.shape[0] * scale_factor)
    original_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    height, width = original_image.shape[:2]
    detection_mask = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    
    # Нормализация изображения
    normalized_image = original_image.astype(np.float32) / 255.0
    
    print("Анализ изображения...")
    print(f"Размер изображения: {width}x{height}")
    print(f"Размер окна: {window_size[0]}x{window_size[1]}")
    print(f"Шаг: {stride}")
    
    total_windows = ((height - window_size[0]) // stride + 1) * ((width - window_size[1]) // stride + 1)
    processed_windows = 0
    
    # Скользящее окно с перекрытием
    for y in range(0, height - window_size[0] + 1, stride):
        for x in range(0, width - window_size[1] + 1, stride):
            # Извлечение участка
            window = normalized_image[y:y + window_size[0], x:x + window_size[1]]
            
            # Изменяем размер окна до размера входа модели
            window_resized = cv2.resize(window, (224, 224))
            
            # Подготовка для модели
            window_batch = np.expand_dims(window_resized, axis=0)
            
            # Получение предсказания
            prediction = model.predict(window_batch, verbose=0)[0][0]
            
            # Если модель уверена в наличии искомого участка
            if prediction > 0.5:  # Порог уверенности
                detection_mask[y:y + window_size[0], x:x + window_size[1]] = 255
                confidence_map[y:y + window_size[0], x:x + window_size[1]] = max(
                    confidence_map[y:y + window_size[0], x:x + window_size[1]].max(),
                    prediction
                )
            
            processed_windows += 1
            if processed_windows % 100 == 0:  # Увеличиваем частоту обновления прогресса
                progress = (processed_windows / total_windows) * 100
                print(f"Прогресс: {progress:.1f}%")
    
    # Обработка маски для объединения близких областей
    kernel = np.ones((3,3), np.uint8)  # Уменьшаем размер ядра
    detection_mask = cv2.morphologyEx(detection_mask, cv2.MORPH_CLOSE, kernel)
    
    # Находим контуры на маске обнаружений
    contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Отрисовка результатов
    result_image = original_image.copy()
    
    # Сначала находим все прямоугольные области
    rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500 and area < 10000:  # Настраиваем размеры областей
            x, y, w, h = cv2.boundingRect(contour)
            # Проверяем соотношение сторон
            aspect_ratio = float(w)/h
            if 0.5 <= aspect_ratio <= 2.0:  # Области не должны быть слишком вытянутыми
                rectangles.append((x, y, w, h))
    
    # Объединяем перекрывающиеся прямоугольники
    merged_rectangles = []
    while rectangles:
        current = rectangles.pop(0)
        x1, y1, w1, h1 = current
        
        i = 0
        while i < len(rectangles):
            x2, y2, w2, h2 = rectangles[i]
            # Проверяем перекрытие
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                # Объединяем прямоугольники
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y
                # Проверяем размер объединенной области
                if w * h < 10000:  # Ограничиваем размер объединенной области
                    current = (x, y, w, h)
                rectangles.pop(i)
            else:
                i += 1
        merged_rectangles.append(current)
    
    print(f"Найдено областей: {len(merged_rectangles)}")
    
    # Отрисовка объединенных прямоугольников
    for rect in merged_rectangles:
        x, y, w, h = rect
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Вычисляем среднюю уверенность для области
        roi_confidence = confidence_map[y:y+h, x:x+w]
        mean_confidence = np.mean(roi_confidence[roi_confidence > 0])
        
        # Добавляем информацию об области
        cv2.putText(
            result_image,
            f'Conf: {mean_confidence:.2f}',
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,  # Уменьшаем размер шрифта
            (0, 255, 0),
            1  # Уменьшаем толщину линий
        )
    
    # Возвращаем изображение в исходный размер
    result_image = cv2.resize(result_image, (int(width/scale_factor), int(height/scale_factor)))
    detection_mask = cv2.resize(detection_mask, (int(width/scale_factor), int(height/scale_factor)))
    
    return result_image, detection_mask

# После обучения модели
model_save_path = 'model/trained_model.h5'
os.makedirs('model', exist_ok=True)
model.save(model_save_path)
print(f"Модель сохранена в {model_save_path}")

# Удаляем ненужные функции и код
def process_single_image(model, image_path):
    pass  # Удаляем эту функцию

def analyze_new_images(model, image_folder):
    pass  # Удаляем эту функцию

if __name__ == "__main__":
    # Оставляем только визуализацию обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Ошибка обучения')
    plt.plot(history.history['val_loss'], label='Ошибка валидации')
    plt.title('График ошибки')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Точность обучения')
    plt.plot(history.history['val_accuracy'], label='Точность валидации')
    plt.title('График точности')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'training_history.png'))
    plt.close()
    
    print("Обучение завершено!")
