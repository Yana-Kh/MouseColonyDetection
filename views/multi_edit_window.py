from PyQt5 import QtWidgets, QtGui, QtCore
import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from database import Database
import sqlite3

class MultiEditWindow(QtWidgets.QWidget):
    """Окно для множественного редактирования областей на изображении"""

    def __init__(self, analysis_dir):
        super().__init__()
        self.analysis_dir = analysis_dir
        self.results_dir = os.path.join(analysis_dir, 'results')
        self.debug_dir = os.path.join(analysis_dir, 'debug_windows')
        
        # Проверяем существование папок
        if not os.path.exists(self.results_dir):
            raise ValueError(f"Папка результатов не найдена: {self.results_dir}")
        
        if not os.path.exists(self.debug_dir):
            raise ValueError(f"Папка {self.debug_dir} не найдена!")
        
        # Инициализируем переменные
        self.current_image_index = 0
        self.current_regions = []
        self.image_files = []
        self.image_dirs = []
        self.current_image = None
        self.edit_mode = "delete"  # Режим по умолчанию - удаление
        self.window_size = 112  # Размер окна для добавления
        
        # Сначала настраиваем UI
        self.setup_ui()
        
        # Затем загружаем изображения
        self.load_images()
        
        # Отмечаем колонии как проверенные только если есть изображения
        if self.image_files:
            try:
                db = Database()
                with sqlite3.connect(db.db_path) as conn:
                    cursor = conn.cursor()
                    # Получаем ID изображения из пути
                    image_path = os.path.join(analysis_dir, self.image_files[self.current_image_index])
                    cursor.execute('SELECT image_id FROM Image WHERE file_path = ?', (image_path,))
                    result = cursor.fetchone()
                    if result:
                        image_id = result[0]
                        # Отмечаем колонии как проверенные
                        cursor.execute('''
                            UPDATE Colony 
                            SET verified = TRUE 
                            WHERE image_id = ?
                        ''', (image_id,))
                        conn.commit()
            except Exception as e:
                print(f"Ошибка при обновлении статуса verified: {str(e)}")
        else:
            QtWidgets.QMessageBox.warning(
                None,
                'Предупреждение',
                'Не найдено изображений для анализа'
            )
            self.close()
        
    def setup_ui(self):
        self.setWindowTitle('Множественное редактирование областей')
        self.setGeometry(100, 100, 800, 600)
        
        # Устанавливаем иконку окна
        icon = QtGui.QIcon('C:/Users/User/PycharmProjects/pythonProject/images/sours/icon.png')
        self.setWindowIcon(icon)
        
        # Белый фон и закругления для окна
        self.setStyleSheet('''
            QWidget {
                background: white;
                border-radius: 15px;
            }
            QLabel {
                border-radius: 8px;
            }
            QGroupBox {
                border-radius: 10px;
            }
            QListWidget {
                background: #E9E9E9;
                border-radius: 10px;
                padding: 5px;
            }
        ''')
        
        # Основной макет
        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)
        
        # Создаем вертикальный контейнер для заголовка и основного содержимого
        main_container = QtWidgets.QVBoxLayout()
        
        # Информация об анализе - теперь над всем содержимым
        info_label = QtWidgets.QLabel(f'Анализ: {os.path.basename(self.analysis_dir)}')
        info_label.setStyleSheet('''
            QLabel {
                font-weight: bold;
                font-size: 14px;
                color: #464646;
                padding: 10px 20px;  /* Увеличили горизонтальный отступ */
                background: white;
                border-bottom: 1px solid #E9E9E9;
                margin: 0;
            }
        ''')
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        main_container.addWidget(info_label)
        
        # Создаем горизонтальный контейнер для левой и правой панели
        content_layout = QtWidgets.QHBoxLayout()
        
        # Левая панель с управлением
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(200)
        
        # Добавляем отступ перед группой режима редактирования
        left_layout.addSpacing(10)
        
        # Выбор режима редактирования
        mode_group = QtWidgets.QGroupBox("Режим редактирования")
        mode_group.setStyleSheet('''
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                padding-top: 20px;
                margin-top: 10px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #464646;
                background: white;
            }
        ''')
        mode_layout = QtWidgets.QVBoxLayout()
        
        self.delete_mode_radio = QtWidgets.QRadioButton("Удаление")
        self.delete_mode_radio.setChecked(True)
        self.delete_mode_radio.toggled.connect(self.change_edit_mode)
        mode_layout.addWidget(self.delete_mode_radio)
        
        self.add_mode_radio = QtWidgets.QRadioButton("Добавление")
        self.add_mode_radio.toggled.connect(self.change_edit_mode)
        mode_layout.addWidget(self.add_mode_radio)
        
        mode_group.setLayout(mode_layout)
        left_layout.addWidget(mode_group)
        
        # Информация о текущем режиме
        self.mode_info_label = QtWidgets.QLabel(
            "Режим: Удаление\nКликните по области для удаления"
        )
        self.mode_info_label.setStyleSheet('''
            QLabel {
                font-style: italic;
                font-size: 12px;
                color: #FF5959;  /* Красный для режима удаления */
            }
        ''')
        self.mode_info_label.setWordWrap(True)
        left_layout.addWidget(self.mode_info_label)
        
        # Добавляем отступ перед группой изображений
        left_layout.addSpacing(10)
        
        # Список изображений
        images_group = QtWidgets.QGroupBox("Изображения")
        images_group.setStyleSheet('''
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                padding-top: 20px;
                margin-top: 10px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #464646;
                background: white;
            }
            QListWidget {
                background: #E9E9E9;
                border-radius: 10px;
                padding: 5px;
            }
        ''')
        images_layout = QtWidgets.QVBoxLayout()
        
        self.images_list = QtWidgets.QListWidget()
        self.images_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.images_list.currentRowChanged.connect(self.change_image)
        images_layout.addWidget(self.images_list)
        
        images_group.setLayout(images_layout)
        left_layout.addWidget(images_group)
        
        # Информация о текущем изображении
        self.image_info_label = QtWidgets.QLabel("Нет данных")
        self.image_info_label.setStyleSheet('font-size: 10px;')
        self.image_info_label.setWordWrap(True)
        left_layout.addWidget(self.image_info_label)
        
        # Кнопка сохранения
        self.save_button = QtWidgets.QPushButton("Сохранить изменения")
        self.save_button.clicked.connect(self.save_changes)
        self.save_button.setStyleSheet('''
            QPushButton {
                background-color: #6EC96E;
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 11px;
                min-width: 100px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #5DB85D;
            }
        ''')
        left_layout.addWidget(self.save_button)
        
        # Кнопка завершения
        self.finish_button = QtWidgets.QPushButton("Завершить редактирование")
        self.finish_button.setStyleSheet('''
            QPushButton {
                background-color: #8AA4BE;
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 11px;
                min-width: 100px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #7B93AB;
            }
        ''')
        self.finish_button.clicked.connect(self.finish_editing)
        left_layout.addWidget(self.finish_button)
        
        left_layout.addStretch()
        
        content_layout.addWidget(left_panel)
        
        # Правая панель с изображением
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Отображение изображения
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        self.image_label = QtWidgets.QLabel("Загрузка изображения...")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.installEventFilter(self)  # Для отслеживания кликов мыши
        
        self.scroll_area.setWidget(self.image_label)
        right_layout.addWidget(self.scroll_area)
        
        content_layout.addWidget(right_panel, stretch=1)
        
        main_container.addLayout(content_layout)
        main_layout.addLayout(main_container)
    
    def load_images(self):
        """Загружает список изображений из папки анализа"""
        try:
            print(f"Загрузка изображений из директории: {self.analysis_dir}")
            
            # Проверяем существование директории
            if not os.path.exists(self.analysis_dir):
                raise ValueError(f"Директория не существует: {self.analysis_dir}")
            
            # Получаем список файлов
            all_files = os.listdir(self.analysis_dir)
            print(f"Всего файлов в директории: {len(all_files)}")
            
            # Ищем оригинальные изображения (не processed_)
            self.image_files = [
                f for f in sorted(all_files)
                if not f.startswith('processed_') and 
                f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            print(f"Найдено изображений: {len(self.image_files)}")
            print("Список изображений:", self.image_files)
            
            if not self.image_files:
                QtWidgets.QMessageBox.warning(
                    None,
                    'Предупреждение',
                    'Не найдено изображений для анализа'
                )
                return
            
            # Создаем список директорий
            self.image_dirs = [
                os.path.splitext(f)[0] for f in self.image_files
            ]
            
            print("Создан список директорий:", self.image_dirs)
            
            # Заполняем список изображений в UI
            if hasattr(self, 'images_list'):
                print("Обновление UI списка изображений")
                self.images_list.clear()
                for filename in self.image_files:
                    self.images_list.addItem(filename)
                
                # Выбираем первое изображение
                if self.images_list.count() > 0:
                    print("Выбор первого изображения")
                    self.images_list.setCurrentRow(0)
                    # Явно вызываем загрузку первого изображения
                    self.change_image(0)
            else:
                print("ОШИБКА: images_list не найден в UI")
            
        except Exception as e:
            print(f"Ошибка при загрузке списка изображений: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(
                None,
                'Ошибка',
                f'Не удалось загрузить список изображений: {str(e)}'
            )
    
    def change_image(self, row):
        """Изменяет текущее отображаемое изображение"""
        print(f"Попытка смены изображения на строку {row}")
        
        if row < 0 or row >= len(self.image_files):
            print(f"Некорректный индекс строки: {row}")
            return
        
        try:
            self.current_image_index = row
            filename = self.image_files[row]
            print(f"Загрузка изображения: {filename}")
            
            # Загружаем оригинальное изображение
            image_path = os.path.join(self.analysis_dir, filename)
            print(f"Полный путь к изображению: {image_path}")
            
            self.current_image = cv2.imread(image_path)
            
            if self.current_image is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                self.image_label.setText(f"Не удалось загрузить изображение {filename}")
                return
            
            print(f"Изображение загружено успешно. Размер: {self.current_image.shape}")
            
            # Загружаем информацию о регионах
            self.load_regions(row)
            
            # Отображаем изображение с регионами
            self.update_image_display()
            
            # Обновляем информацию
            self.image_info_label.setText(
                f"Изображение: {filename}\n"
                f"Размер: {self.current_image.shape[1]}x{self.current_image.shape[0]}\n"
                f"Найдено областей: {len(self.current_regions)}"
            )
            
        except Exception as e:
            print(f"Ошибка при смене изображения: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(
                None,
                'Ошибка',
                f'Не удалось загрузить изображение: {str(e)}'
            )
    
    def load_regions(self, image_index):
        """Загружает регионы для указанного изображения"""
        if not self.image_files or image_index < 0 or image_index >= len(self.image_files):
            print("Некорректный индекс изображения или список изображений пуст")
            return False
        
        try:
            # Получаем путь к текущему изображению
            image_path = os.path.join(self.analysis_dir, self.image_files[image_index])
            
            # Подключаемся к базе данных
            db = Database()
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            
            # Получаем image_id
            cursor.execute('SELECT image_id FROM Image WHERE file_path = ?', (image_path,))
            result = cursor.fetchone()
            
            if not result:
                print(f"Изображение не найдено в базе данных: {image_path}")
                return False
            
            image_id = result[0]
            
            # Получаем все колонии для этого изображения
            cursor.execute('''
                SELECT window_path, x, y, confidence
                FROM Colony 
                WHERE image_id = ?
                ORDER BY confidence DESC
            ''', (image_id,))
            
            colonies = cursor.fetchall()
            
            # Очищаем текущие регионы
            self.current_regions = []
            
            # Преобразуем данные в формат регионов
            for window_path, x, y, confidence in colonies:
                if os.path.exists(window_path):
                    self.current_regions.append({
                        'x': x,
                        'y': y,
                        'prediction': confidence,
                        'window_path': window_path
                    })
            
            # Обновляем UI
            if hasattr(self, 'image_info_label'):
                self.image_info_label.setText(
                    f'Изображение {image_index + 1} из {len(self.image_files)}\n'
                    f'Найдено регионов: {len(self.current_regions)}'
                )
            
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке регионов: {str(e)}")
            QtWidgets.QMessageBox.warning(
                None,
                'Ошибка',
                f'Не удалось загрузить регионы: {str(e)}'
            )
            return False
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    def update_image_display(self):
        """Обновляет отображение изображения с выделенными областями"""
        if self.current_image is None:
            return
        
        # Создаем копию изображения
        display_image = self.current_image.copy()
        
        print(f"Отображение {len(self.current_regions)} регионов")
        print("ПРОВЕРКА РЕГИОНОВ ПЕРЕД ОТРИСОВКОЙ:")
        for i, region in enumerate(self.current_regions):
            x, y = region['x'], region['y']
            print(f"Отрисовка региона {i+1}: x={x}, y={y}, prediction={region['prediction']}")
            
            # Проверка на некорректные координаты
            if x == 505 and y == 505:
                print(f"ОБНАРУЖЕН ПРОБЛЕМНЫЙ РЕГИОН С КООРДИНАТАМИ (505, 505)!")
                print(f"Полная информация: {region}")
            
            # Проверяем границы изображения
            if (x < 0 or y < 0 or 
                x + self.window_size > self.current_image.shape[1] or 
                y + self.window_size > self.current_image.shape[0]):
                print(f"ПРЕДУПРЕЖДЕНИЕ: Регион выходит за границы изображения!")
                print(f"Размер изображения: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
                continue  # Пропускаем отрисовку этого региона
            
            # Рисуем прямоугольник с яркими цветами и большей толщиной
            cv2.rectangle(
                display_image,
                (x, y),
                (x + self.window_size, y + self.window_size),
                (0, 0, 255),  # Красный цвет
                3  # Увеличенная толщина
            )
            
            # Добавим заполнение с прозрачностью для лучшей видимости
            overlay = display_image.copy()
            cv2.rectangle(
                overlay,
                (x, y),
                (x + self.window_size, y + self.window_size),
                (0, 0, 255),  # Красный цвет
                -1  # Заполненный прямоугольник
            )
            # Наложение с прозрачностью
            cv2.addWeighted(overlay, 0.2, display_image, 0.8, 0, display_image)
            
            # Добавляем текст с координатами
            center_x = x + self.window_size // 2
            center_y = y + self.window_size // 2
            
            # Рисуем белый текст с черной обводкой для лучшей видимости
            text = f"({center_x},{center_y}) {region['prediction']:.2f}"
            cv2.putText(
                display_image,
                text,
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Больший размер шрифта
                (0, 0, 0),  # Черный контур
                2
            )
            cv2.putText(
                display_image,
                text,
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # Белый текст
                1
            )
        
        # ЗАТЕМ масштабируем изображение для отображения
        height, width = display_image.shape[:2]
        max_size = 800  # Максимальный размер по любой стороне
        
        # Вычисляем коэффициент масштабирования
        scale = min(max_size / width, max_size / height)
        if scale < 1:  # Масштабируем только если изображение больше max_size
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)
            print(f"Изображение уменьшено до {new_width}x{new_height}, масштаб={scale}")
        
        # Конвертируем в формат для Qt
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        
        # Убедимся, что изображение в правильном формате
        if display_image.ndim == 3 and display_image.shape[2] == 3:
            # BGR -> RGB для Qt
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            
            q_image = QtGui.QImage(
                display_image_rgb.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format_RGB888
            )
        else:
            # Черно-белое изображение
            q_image = QtGui.QImage(
                display_image.data,
                width,
                height,
                width,
                QtGui.QImage.Format_Grayscale8
            )
        
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(width, height)
    
    def change_edit_mode(self):
        """Изменяет режим редактирования"""
        if self.delete_mode_radio.isChecked():
            self.edit_mode = "delete"
            self.mode_info_label.setStyleSheet('''
                QLabel {
                    font-style: italic;
                    font-size: 12px;
                    color: #FF5959;
                }
            ''')
            self.mode_info_label.setText("Режим: Удаление\nКликните по области для удаления")
        else:
            self.edit_mode = "add"
            self.mode_info_label.setStyleSheet('''
                QLabel {
                    font-style: italic;
                    font-size: 12px;
                    color: #6EC96E;
                }
            ''')
            self.mode_info_label.setText("Режим: Добавление\nКликните левой кнопкой мыши для добавления новой области")
    
    def eventFilter(self, obj, event):
        """Обрабатывает события щелчков мыши на изображении"""
        if obj is self.image_label and self.current_image is not None:
            # Обработка клика левой кнопкой мыши (удаление или добавление)
            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                point = event.pos()
                if self.edit_mode == "delete":
                    self.handle_delete_click(point.x(), point.y())
                else:
                    self.handle_add_click(point.x(), point.y())
                return True
                
        return super().eventFilter(obj, event)
    
    def handle_delete_click(self, x, y):
        """Обрабатывает клик для удаления области"""
        if self.current_image is None:
            return
        
        # Получаем масштаб текущего отображения
        height, width = self.current_image.shape[:2]
        max_size = 800
        scale = min(max_size / width, max_size / height)
        if scale >= 1:
            scale = 1.0
        
        # Преобразуем координаты клика обратно в координаты исходного изображения
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        
        print(f"Клик для удаления в точке ({x}, {y}), в исходных координатах ({orig_x}, {orig_y})")
        
        # Ищем область, которая содержит точку клика
        for i, region in enumerate(self.current_regions):
            rx, ry = region['x'], region['y']
            # Проверяем, попадает ли клик в прямоугольник области
            if rx <= orig_x <= rx + self.window_size and ry <= orig_y <= ry + self.window_size:
                try:
                    print(f"Найдена область для удаления: {region}")
                    
                    # Удаляем файл окна
                    if os.path.exists(region['window_path']):
                        os.remove(region['window_path'])
                        print(f"Удален файл: {region['window_path']}")
                    else:
                        print(f"Файл не найден: {region['window_path']}")
                    
                    # Удаляем из списка регионов
                    deleted_region = self.current_regions.pop(i)
                    
                    # Обновляем отображение
                    self.update_image_display()
                    
                    # Обновляем информацию
                    original_filename = self.image_files[self.current_image_index]
                    self.image_info_label.setText(
                        f"Изображение: {original_filename}\n"
                        f"Размер: {self.current_image.shape[1]}x{self.current_image.shape[0]}\n"
                        f"Найдено областей: {len(self.current_regions)}\n"
                        f"Удалена область: x={rx}, y={ry} (уверенность: {deleted_region['prediction']:.2f})"
                    )
                    
                    return
                    
                except Exception as e:
                    print(f"Ошибка при удалении: {str(e)}")
                    QtWidgets.QMessageBox.warning(
                        None,
                        'Ошибка',
                        f'Не удалось удалить область: {str(e)}'
                    )
                    return
        
        # Если клик не попал ни в одну область - тоже убираем всплывающее сообщение
        print("Область не найдена для координат")
    
    def handle_add_click(self, x, y):
        """Обрабатывает клик для добавления новой области"""
        if self.current_image is None:
            return
        
        # Получаем масштаб текущего отображения
        height, width = self.current_image.shape[:2]
        max_size = 800
        scale = min(max_size / width, max_size / height)
        if scale >= 1:
            scale = 1.0
        
        # Преобразуем координаты клика обратно в координаты исходного изображения
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        
        print(f"Клик для добавления в точке ({x}, {y}), в исходных координатах ({orig_x}, {orig_y})")
        
        try:
            # Вычисляем центр нового окна в исходных координатах
            center_x = orig_x
            center_y = orig_y
            
            # Вычисляем левый верхний угол окна (центр минус половина размера)
            top_left_x = center_x - self.window_size // 2
            top_left_y = center_y - self.window_size // 2
            
            print(f"Верхний левый угол окна: ({top_left_x}, {top_left_y})")
            
            # Проверяем, не выходит ли окно за границы изображения
            if (top_left_x < 0 or top_left_y < 0 or 
                top_left_x + self.window_size > self.current_image.shape[1] or 
                top_left_y + self.window_size > self.current_image.shape[0]):
                QtWidgets.QMessageBox.warning(
                    None,
                    'Предупреждение',
                    'Окно выходит за границы изображения'
                )
                print("Окно выходит за границы изображения")
                return
            
            # Извлекаем окно из текущего изображения
            window = self.current_image[
                top_left_y:top_left_y + self.window_size,
                top_left_x:top_left_x + self.window_size
            ]
            
            # Проверяем, что окно не пустое
            if window.size == 0:
                QtWidgets.QMessageBox.warning(
                    None,
                    'Ошибка',
                    'Не удалось извлечь окно из изображения'
                )
                print(f"Пустое окно: размер={window.shape}")
                return
            
            # Генерируем имя для нового окна
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            window_filename = f"manual_window_{timestamp}_x{top_left_x}_y{top_left_y}_pred1.000.png"
            
            # Получаем имя исходного файла и папку debug_windows
            original_filename = self.image_files[self.current_image_index]
            image_name = os.path.splitext(original_filename)[0]
            image_debug_dir = os.path.join(self.debug_dir, image_name)
            
            # Создаем папку, если не существует
            os.makedirs(image_debug_dir, exist_ok=True)
            
            # Путь для сохранения окна
            window_path = os.path.join(image_debug_dir, window_filename)
            
            # Сохраняем окно
            print(f"Сохраняем окно по пути: {window_path}")
            cv2.imwrite(window_path, window)
            
            # Добавляем в список регионов
            self.current_regions.append({
                'x': top_left_x,
                'y': top_left_y,
                'prediction': 1.0,  # Максимальная уверенность для ручного добавления
                'filename': window_filename,
                'window_path': window_path
            })
            
            # Обновляем отображение
            self.update_image_display()
            
            # Обновляем информацию
            self.image_info_label.setText(
                f"Изображение: {original_filename}\n"
                f"Размер: {self.current_image.shape[1]}x{self.current_image.shape[0]}\n"
                f"Найдено областей: {len(self.current_regions)}\n"
                f"Добавлена область: x={top_left_x}, y={top_left_y}"
            )
            
            print(f"Добавлена новая область: x={top_left_x}, y={top_left_y}")
            
        except Exception as e:
            print(f"Ошибка при добавлении: {str(e)}")
            QtWidgets.QMessageBox.warning(
                None,
                'Ошибка',
                f'Не удалось добавить область: {str(e)}'
            )
    
    def save_changes(self):
        """Сохраняет изменения в базу данных"""
        if not self.current_regions:
            return
        
        try:
            # Получаем путь к текущему изображению
            filename = self.image_files[self.current_image_index]
            image_path = os.path.join(self.analysis_dir, filename)
            
            # Подключаемся к базе данных
            db = Database()
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            
            # Получаем image_id
            cursor.execute('SELECT image_id FROM Image WHERE file_path = ?', (image_path,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Изображение не найдено в базе данных: {image_path}")
            
            image_id = result[0]
            
            # Удаляем все существующие колонии для этого изображения
            cursor.execute('DELETE FROM Colony WHERE image_id = ?', (image_id,))
            
            # Получаем размеры изображения
            img_width = self.current_image.shape[1]
            img_height = self.current_image.shape[0]
            print(f"Размеры текущего изображения: {img_width}x{img_height}")
            
            # Добавляем обновленные данные
            for region in self.current_regions:
                # Проверяем границы изображения
                if (region['x'] < 0 or region['y'] < 0 or 
                    region['x'] + self.window_size > img_width or 
                    region['y'] + self.window_size > img_height):
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Регион выходит за границы изображения: x={region['x']}, y={region['y']}")
                    # Корректируем координаты
                    region['x'] = max(0, min(region['x'], img_width - self.window_size))
                    region['y'] = max(0, min(region['y'], img_height - self.window_size))
                    print(f"Скорректированные координаты: x={region['x']}, y={region['y']}")
                
                # Вычисляем центр квадрата
                center_x = region['x'] + self.window_size // 2
                center_y = region['y'] + self.window_size // 2
                
                # Добавляем колонию в базу данных
                cursor.execute('''
                    INSERT INTO Colony (
                        image_id, x, y, confidence, window_path, 
                        latitude, longitude, verified
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id,
                    region['x'],
                    region['y'],
                    region['prediction'],
                    region['window_path'],
                    0.0,  # latitude - нужно будет добавить пересчет координат
                    0.0,  # longitude - нужно будет добавить пересчет координат
                    True  # verified - так как это ручное редактирование
                ))
            
            # Сохраняем изменения
            conn.commit()
            
            # Обновляем изображение результатов
            self.update_result_image(os.path.splitext(filename)[0], filename)
            
            QtWidgets.QMessageBox.information(
                None,
                'Успех',
                'Изменения сохранены в базу данных'
            )
            
        except Exception as e:
            print(f"Ошибка при сохранении изменений: {str(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(
                None,
                'Ошибка',
                f'Не удалось сохранить изменения: {str(e)}'
            )
            
            # Откатываем изменения в случае ошибки
            if 'conn' in locals():
                conn.rollback()
        
        finally:
            # Закрываем соединение
            if 'conn' in locals():
                conn.close()
    
    def update_result_image(self, image_name, original_filename):
        """Обновляет изображение результатов с новыми областями"""
        # Загружаем исходное изображение
        original_path = os.path.join(self.analysis_dir, original_filename)
        if not os.path.exists(original_path):
            print(f"Не найдено исходное изображение: {original_path}")
            return
        
        original_image = cv2.imread(original_path)
        if original_image is None:
            print(f"Не удалось загрузить исходное изображение: {original_path}")
            return
        
        # Создаем копию для результатов
        result_image = original_image.copy()
        
        # Рисуем области
        for region in self.current_regions:
            x, y = region['x'], region['y']
            center_x = x + self.window_size // 2
            center_y = y + self.window_size // 2
            
            # Рисуем прямоугольник
            cv2.rectangle(
                result_image,
                (x, y),
                (x + self.window_size, y + self.window_size),
                (0, 255, 0),
                2
            )
            
            # Добавляем текст с координатами центра
            cv2.putText(
                result_image,
                f"({center_x},{center_y})",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # Сохраняем обновленное изображение результатов
        result_path = os.path.join(self.results_dir, f'processed_{original_filename}')
        print(f"Сохраняем обновленное изображение результатов: {result_path}")
        cv2.imwrite(result_path, result_image)
    
    def finish_editing(self):
        """Завершение редактирования и возврат к окну выбора режима"""
        # Сначала сохраняем любые несохраненные изменения
        try:
            self.save_changes()
        except Exception as e:
            print(f"Ошибка при сохранении изменений: {str(e)}")
        
        self.hide()
        
        # Показываем диалог выбора режима редактирования
        from edit_mode_dialog import EditModeDialog
        self.edit_mode_dialog = EditModeDialog(self.analysis_dir)  # Исправлено: передаем только один параметр
        self.edit_mode_dialog.exec_() 