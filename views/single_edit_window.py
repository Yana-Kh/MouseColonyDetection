from PyQt5 import QtWidgets, QtGui, QtCore
import os
import pandas as pd
import shutil
from database import Database
import sqlite3

class SingleEditWindow(QtWidgets.QWidget):
    """Окно для редактирования отдельных окон анализа"""

    def __init__(self, analysis_dir):
        super().__init__()
        self.analysis_dir = analysis_dir
        
        # Проверяем существование папки debug_windows
        self.debug_dir = os.path.join(analysis_dir, 'debug_windows')
        if not os.path.exists(self.debug_dir):
            raise ValueError(f"Папка {self.debug_dir} не найдена!")
        
        print(f"Инициализация окна редактирования для {analysis_dir}")
        print(f"Папка debug_windows: {self.debug_dir}")
        
        # Инициализируем все необходимые переменные
        self.current_image_dir = None
        self.current_image_index = 0
        self.current_window_index = 0
        self.image_dirs = []
        self.windows = []
        self.image_files = []
        
        # Сначала загружаем только список файлов
        self.load_image_files()
        
        # Настраиваем UI
        self.setup_ui()
        
        # Теперь загружаем окна для первого изображения
        if self.image_files:
            self.load_current_image_windows(0)
            
            # Отмечаем колонии как проверенные
            try:
                db = Database()
                conn = sqlite3.connect(db.db_path)
                cursor = conn.cursor()
                
                image_path = os.path.join(analysis_dir, self.image_files[self.current_image_index])
                cursor.execute('SELECT image_id FROM Image WHERE file_path = ?', (image_path,))
                result = cursor.fetchone()
                if result:
                    image_id = result[0]
                    cursor.execute('''
                        UPDATE Colony 
                        SET verified = TRUE 
                        WHERE image_id = ?
                    ''', (image_id,))
                    conn.commit()
                
            except Exception as e:
                print(f"Ошибка при обновлении статуса verified: {str(e)}")
            finally:
                if 'conn' in locals():
                    conn.close()
    
    def setup_ui(self):
        self.setWindowTitle('Просмотр и редактирование окон анализа')
        self.setGeometry(100, 100, 600, 450)
        
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
        ''')
        
        # Основной макет
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)
        
        # Информация об анализе - над всем содержимым
        info_label = QtWidgets.QLabel(f'Анализ: {os.path.basename(self.analysis_dir)}')
        info_label.setStyleSheet('''
            QLabel {
                font-weight: bold;
                font-size: 14px;
                color: #464646;
                padding: 10px;
                background: white;
                border-bottom: 1px solid #E9E9E9;
                margin: 0;
            }
        ''')
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(info_label)
        main_layout.setContentsMargins(0, 0, 0, 10)  # Убираем отступы по бокам
        
        # Информация о текущем изображении
        self.image_info_label = QtWidgets.QLabel('Загрузка...')
        self.image_info_label.setStyleSheet('font-size: 11px;')
        main_layout.addWidget(self.image_info_label)
        
        # Контейнер для изображения
        image_container = QtWidgets.QWidget()
        image_layout = QtWidgets.QHBoxLayout()
        image_container.setLayout(image_layout)
        
        # Отображение изображения
        self.image_label = QtWidgets.QLabel('Загрузка изображения...')
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setMaximumSize(300, 300)
        self.image_label.setStyleSheet('border: 1px solid #ccc;')
        image_layout.addWidget(self.image_label)
        
        main_layout.addWidget(image_container)
        
        # Информация об окне
        self.window_info_label = QtWidgets.QLabel('Информация об окне')
        self.window_info_label.setStyleSheet('font-size: 11px;')
        main_layout.addWidget(self.window_info_label)
        
        # Кнопки навигации между окнами
        nav_layout = QtWidgets.QHBoxLayout()
        
        self.prev_button = QtWidgets.QPushButton('← Пред. окно')
        self.prev_button.setStyleSheet(nav_button_style)
        self.prev_button.clicked.connect(self.show_previous_window)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QtWidgets.QPushButton('След. окно →')
        self.next_button.setStyleSheet(nav_button_style)
        self.next_button.clicked.connect(self.show_next_window)
        nav_layout.addWidget(self.next_button)
        
        main_layout.addLayout(nav_layout)
        
        # Кнопки действий
        actions_layout = QtWidgets.QHBoxLayout()
        
        self.colony_button = QtWidgets.QPushButton('Есть колония ✓')
        self.colony_button.setStyleSheet(colony_button_style)
        self.colony_button.clicked.connect(self.confirm_colony)
        actions_layout.addWidget(self.colony_button)
        
        self.no_colony_button = QtWidgets.QPushButton('Нет колонии ✗')
        self.no_colony_button.setStyleSheet(no_colony_button_style)
        self.no_colony_button.clicked.connect(self.reject_colony)
        actions_layout.addWidget(self.no_colony_button)
        
        main_layout.addLayout(actions_layout)
        
        # Кнопка завершения
        self.finish_button = QtWidgets.QPushButton('Завершить')
        self.finish_button.setStyleSheet('''
            QPushButton {
                background-color: #8AA4BE;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 17px;
                font-size: 13px;
                min-width: 160px;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #7B93AB;
            }
        ''')
        self.finish_button.clicked.connect(self.finish_editing)
        main_layout.addWidget(self.finish_button)
    
    def load_image_files(self):
        """Загружает только список файлов изображений"""
        # Ищем оригинальные изображения (не processed_)
        self.image_files = [
            f for f in sorted(os.listdir(self.analysis_dir))
            if not f.startswith('processed_') and 
            f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not self.image_files:
            # Используем стандартное окно уведомления
            QtWidgets.QMessageBox.warning(
                None,  # Убираем self как родителя
                'Предупреждение',
                'Не найдено изображений для анализа'
            )
            return
        
        # Создаем список директорий для совместимости
        self.image_dirs = [
            os.path.splitext(f)[0] for f in self.image_files
        ]
    
    def load_current_image_windows(self, index):
        """Загружает окна анализа для текущего изображения из базы данных"""
        if not self.image_dirs or index < 0 or index >= len(self.image_dirs):
            return False
        
        self.current_image_index = index
        self.current_image_dir = os.path.join(self.debug_dir, self.image_dirs[index])
        
        try:
            # Подключаемся к базе данных
            db = Database()
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            
            # Получаем image_id для текущего изображения
            image_path = os.path.join(self.analysis_dir, self.image_files[index])
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
            
            # Преобразуем в формат для окон
            self.windows = []
            for window_path, x, y, confidence in colonies:
                if os.path.exists(window_path):
                    self.windows.append({
                        'path': window_path,
                        'x': x,
                        'y': y,
                        'prediction': confidence,
                        'filename': os.path.basename(window_path)
                    })
            
            # Обновляем информацию
            self.image_info_label.setText(
                f'Изображение {index+1} из {len(self.image_dirs)}: {self.image_dirs[index]}\n'
                f'Найдено окон: {len(self.windows)}'
            )
            
            # Показываем первое окно
            self.current_window_index = 0
            if self.windows:
                self.show_current_window()
                return True
            else:
                self.image_label.setText('Нет окон для отображения')
                self.window_info_label.setText('Нет данных')
                return False
            
        except Exception as e:
            print(f"Ошибка при загрузке окон: {str(e)}")
            QtWidgets.QMessageBox.warning(
                None,  # Убираем self как родителя
                'Ошибка',
                f'Не удалось загрузить окна: {str(e)}'
            )
            return False
        
        finally:
            if 'conn' in locals():
                conn.close()
        
        # Обновляем состояние кнопок навигации между изображениями
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.image_dirs) - 1)
    
    def show_current_window(self):
        """Отображает текущее окно анализа"""
        if not self.windows or self.current_window_index >= len(self.windows):
            return
        
        window = self.windows[self.current_window_index]
        
        # Загружаем изображение
        pixmap = QtGui.QPixmap(window['path'])
        if not pixmap.isNull():
            # Масштабируем изображение, сохраняя пропорции
            scaled_pixmap = pixmap.scaled(
                300, 300,  # Фиксированный размер для масштабирования
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText('Ошибка загрузки изображения')
        
        # Обновляем информацию об окне
        self.window_info_label.setText(
            f'Окно {self.current_window_index+1} из {len(self.windows)} | '
            f'X={window["x"]}, Y={window["y"]} | '
            f'Уверенность: {window["prediction"]:.3f}'
        )
        
        # Обновляем состояние кнопок
        self.prev_button.setEnabled(self.current_window_index > 0)
        self.next_button.setEnabled(self.current_window_index < len(self.windows) - 1)
    
    def show_next_window(self):
        """Показывает следующее окно"""
        if self.current_window_index < len(self.windows) - 1:
            self.current_window_index += 1
            self.show_current_window()
    
    def show_previous_window(self):
        """Показывает предыдущее окно"""
        if self.current_window_index > 0:
            self.current_window_index -= 1
            self.show_current_window()
    
    def confirm_colony(self):
        """Подтверждает наличие колонии в текущем окне"""
        if self.current_window_index == len(self.windows) - 1:
            # Если это последнее окно текущего изображения, переходим к следующему изображению
            if self.current_image_index < len(self.image_dirs) - 1:
                self.show_next_image()
        else:
            # Иначе переходим к следующему окну
            self.show_next_window()
    
    def reject_colony(self):
        """Отклоняет наличие колонии в текущем окне"""
        if not self.windows or self.current_window_index >= len(self.windows):
            return
        
        window = self.windows[self.current_window_index]
        
        try:
            # Удаляем файл окна
            if os.path.exists(window['path']):
                os.remove(window['path'])
                print(f"Удален файл: {window['path']}")
            
            # Удаляем запись из базы данных
            db = Database()
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            
            # Получаем image_id
            image_path = os.path.join(self.analysis_dir, self.image_dirs[self.current_image_index])
            cursor.execute('SELECT image_id FROM Image WHERE file_path = ?', (image_path,))
            result = cursor.fetchone()
            if result:
                image_id = result[0]
                # Удаляем колонию по координатам и image_id
                cursor.execute('''
                    DELETE FROM Colony 
                    WHERE image_id = ? AND x = ? AND y = ?
                ''', (image_id, window['x'], window['y']))
                conn.commit()
            
            # Удаляем из списка окон
            self.windows.pop(self.current_window_index)
            
            # Если текущий индекс теперь за пределами списка, корректируем его
            if self.current_window_index >= len(self.windows):
                self.current_window_index = max(0, len(self.windows) - 1)
            
            # Обновляем отображение
            if self.windows:
                self.show_current_window()
            else:
                self.image_label.setText('Нет окон для отображения')
                self.window_info_label.setText('Нет данных')
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
            
            # Обновляем информацию об изображении
            self.image_info_label.setText(
                f'Изображение {self.current_image_index+1} из {len(self.image_dirs)}: {self.image_dirs[self.current_image_index]}\n'
                f'Найдено окон: {len(self.windows)}'
            )
            
        except Exception as e:
            print(f"Ошибка при удалении колонии: {str(e)}")
            QtWidgets.QMessageBox.warning(
                None,  # Убираем self как родителя
                'Ошибка',
                f'Не удалось удалить колонию: {str(e)}'
            )
        finally:
            if 'conn' in locals():
                conn.close()
    
    def show_next_image(self):
        """Переходит к следующему изображению"""
        if self.current_image_index < len(self.image_dirs) - 1:
            self.load_current_image_windows(self.current_image_index + 1)
    
    def show_previous_image(self):
        """Переходит к предыдущему изображению"""
        if self.current_image_index > 0:
            self.load_current_image_windows(self.current_image_index - 1)
    
    def finish_editing(self):
        """Завершение редактирования и возврат к окну выбора режима"""
        self.hide()
        
        # Показываем диалог выбора режима редактирования
        from edit_mode_dialog import EditModeDialog
        self.edit_mode_dialog = EditModeDialog(self.analysis_dir)
        self.edit_mode_dialog.exec_()

# Общий стиль для навигационных кнопок
nav_button_style = '''
    QPushButton {
        background: #8AA4BE;
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
'''

# Стиль для кнопки "Есть колония"
colony_button_style = '''
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
'''

# Стиль для кнопки "Нет колонии"
no_colony_button_style = '''
    QPushButton {
        background-color: #FF5959;
        color: white;
        border: none;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        min-width: 100px;
        min-height: 22px;
    }
    QPushButton:hover {
        background-color: #E65252;
    }
''' 