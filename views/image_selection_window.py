# import sys
# sys.path('C:/Users/User/PycharmProjects')

from PyQt5 import QtWidgets, QtGui, QtCore
import os
import shutil
from datetime import datetime
from analyze_image import analyze_new_images, analyze_images
from tensorflow.keras.models import load_model
from edit_mode_dialog import EditModeDialog

class ImageSelectionWindow(QtWidgets.QWidget):
    def __init__(self, image_files):
        super().__init__()
        self.image_files = image_files
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle('Выбранные изображения')
        self.setGeometry(200, 200, 600, 400)
        
        # Устанавливаем белый фон и закругленные углы для всего окна
        self.setStyleSheet('''
            QWidget {
                background: white;
                border-radius: 30px;
            }
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 15px;
                padding: 5px;
            }
            QListWidget::item {
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item:selected {
                background: #f0f0f0;
                color: #464646;
            }
        ''')

        # Основной макет
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Горизонтальный макет для списка и кнопок редактирования
        list_edit_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(list_edit_layout)

        # Список изображений
        self.list_widget = QtWidgets.QListWidget()
        for image_file in self.image_files:
            item = QtWidgets.QListWidgetItem(os.path.basename(image_file))
            item.setData(QtCore.Qt.UserRole, image_file)  # Сохраняем полный путь
            self.list_widget.addItem(item)
        list_edit_layout.addWidget(self.list_widget)

        # Макет для кнопок редактирования (вертикальный)
        edit_buttons_layout = QtWidgets.QVBoxLayout()
        edit_buttons_layout.setAlignment(QtCore.Qt.AlignTop)
        
        # Стиль для кнопок редактирования
        edit_button_style = '''
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 10px;
                font-size: 12px;
                min-width: 80px;
                max-width: 80px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#delete_button {
                background-color: #FF5959;
            }
            QPushButton#delete_button:hover {
                background-color: #E65252;
            }
            QPushButton#add_button {
                background-color: #6EC96E;
            }
            QPushButton#add_button:hover {
                background-color: #5DB85D;
            }
        '''

        # Стиль для основных кнопок
        main_button_style = '''
            QPushButton {
                background-color: #6EC96E;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 15px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5DB85D;
            }
            QPushButton#cancel_button {
                background-color: #FF5959;
            }
            QPushButton#cancel_button:hover {
                background-color: #E65252;
            }
        '''

        # Кнопка добавления
        self.add_button = QtWidgets.QPushButton('Добавить')
        self.add_button.setObjectName('add_button')
        self.add_button.setStyleSheet(edit_button_style)
        self.add_button.clicked.connect(self.add_images)
        edit_buttons_layout.addWidget(self.add_button)

        # Кнопка удаления
        self.delete_button = QtWidgets.QPushButton('Удалить')
        self.delete_button.setObjectName('delete_button')
        self.delete_button.setStyleSheet(edit_button_style)
        self.delete_button.clicked.connect(self.delete_selected)
        edit_buttons_layout.addWidget(self.delete_button)

        # Добавляем отступ между кнопками редактирования и списком
        edit_buttons_layout.setContentsMargins(10, 0, 0, 0)
        list_edit_layout.addLayout(edit_buttons_layout)

        # Макет для основных кнопок
        button_layout = QtWidgets.QHBoxLayout()

        # Кнопка отмены
        self.cancel_button = QtWidgets.QPushButton('Отмена')
        self.cancel_button.setObjectName('cancel_button')
        self.cancel_button.setStyleSheet(main_button_style)
        self.cancel_button.clicked.connect(self.hide)
        button_layout.addWidget(self.cancel_button)

        # Кнопка начала анализа
        self.analyze_button = QtWidgets.QPushButton('Начать анализ')
        self.analyze_button.setStyleSheet(main_button_style)
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)

        layout.addLayout(button_layout)

    def add_images(self):
        file_dialog = QtWidgets.QFileDialog()
        files, _ = file_dialog.getOpenFileNames(
            self,
            'Выберите изображения',
            '',
            'Изображения (*.png *.jpg *.jpeg *.bmp *.gif);;Все файлы (*.*)'
        )
        
        for file_path in files:
            # Проверяем, нет ли уже такого файла в списке
            exists = False
            for i in range(self.list_widget.count()):
                if self.list_widget.item(i).data(QtCore.Qt.UserRole) == file_path:
                    exists = True
                    break
            
            if not exists:
                item = QtWidgets.QListWidgetItem(os.path.basename(file_path))
                item.setData(QtCore.Qt.UserRole, file_path)
                self.list_widget.addItem(item)

    def delete_selected(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(
                None,
                'Предупреждение',
                'Пожалуйста, выберите файлы для удаления'
            )
            return

        for item in selected_items:
            self.list_widget.takeItem(self.list_widget.row(item))

    def start_analysis(self):
        """Запускает анализ выбранных изображений"""
        try:
            # Создаем основную папку для всех анализов
            analysis_dir = 'analysis'
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Создаем папку с датой и временем внутри папки analysis
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            images_dir = os.path.join(analysis_dir, f'analysis_{timestamp}')
            os.makedirs(images_dir, exist_ok=True)
            
            # Копируем выбранные файлы в новую папку
            copied_files = []
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                source_path = item.data(QtCore.Qt.UserRole)
                filename = os.path.basename(source_path)
                destination_path = os.path.join(images_dir, filename)
                
                try:
                    shutil.copy2(source_path, destination_path)
                    copied_files.append(destination_path)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        None,
                        'Ошибка',
                        f'Не удалось скопировать файл {filename}: {str(e)}'
                    )
                    return
            
            # Анализируем все изображения в рамках одного анализа
            analysis_id = analyze_images(copied_files, os.path.join(images_dir, 'debug_windows'))
            
            if analysis_id is not None:
                QtWidgets.QMessageBox.information(
                    None,
                    'Успех',
                    f'Анализ завершен. Результаты сохранены в папке {images_dir}'
                )
                
                # Показываем диалог выбора режима редактирования
                self.edit_dialog = EditModeDialog(images_dir)
                self.edit_dialog.exec_()
            else:
                QtWidgets.QMessageBox.warning(
                    None,
                    'Ошибка',
                    'Не удалось выполнить анализ изображений'
                )
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                'Ошибка',
                f'Ошибка при анализе изображений: {str(e)}'
            )

            # Создаем информационное окно с успешным завершением
            msg = QtWidgets.QMessageBox(None)
            msg.setWindowTitle("Успех")
            msg.setText("Анализ завершен успешно!")
            msg.setIcon(QtWidgets.QMessageBox.Information)
            
            # Настраиваем стиль окна
            msg.setStyleSheet('''
                QMessageBox {
                    background: white;
                    border-radius: 15px;
                }
                QLabel {
                    color: #464646;
                    font-size: 14px;
                    min-width: 300px;
                }
                QPushButton {
                    background-color: #6EC96E;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 15px;
                    min-width: 80px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #5DB85D;
                }
            ''') 