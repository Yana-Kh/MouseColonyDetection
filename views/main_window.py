from PyQt5 import QtWidgets, QtGui, QtCore
import os
import shutil
from image_selection_window import ImageSelectionWindow
from report_viewer_window import ReportViewerWindow
from PyQt5.QtGui import QIcon

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Classification App')
        self.setFixedSize(800, 400)  # Фиксированный размер окна
        
        # Создание виджетов
        self.create_widgets()

    def create_widgets(self):
        # Центральный виджет
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet('background-color: white;')

        # Основной макет без отступов для градиента
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # Убираем отступы
        central_widget.setLayout(main_layout)

        # Левый макет
        left_widget = QtWidgets.QWidget()
        left_widget.setContentsMargins(30, 30, 30, 30)  # Отступы только для левой части
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        main_layout.addWidget(left_widget, stretch=1)

        # Приветствие большими буквами
        welcome_label = QtWidgets.QLabel('ДОБРО ПОЖАЛОВАТЬ')
        welcome_label.setStyleSheet('''
            QLabel {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
                font-family: Roboto;
            }
        ''')
        left_layout.addWidget(welcome_label)

        # Описание приложения
        description_label = QtWidgets.QLabel(
            'Это приложение предназначено для классификации изображений\n'
            'и создания отчетов на основе анализа.'
        )
        description_label.setStyleSheet('''
            QLabel {
                font-size: 14px;
                font-family: Roboto;
                background: white;
                color: #464646;
                padding: 10px;
                border-radius: 30px;
            }
        ''')
        description_label.setWordWrap(True)
        left_layout.addWidget(description_label)

        # Описание кнопок
        buttons_description = QtWidgets.QLabel(
            '\nКнопка "Загрузить фото" позволяет выбрать изображения '
            'для анализа и классификации.\n\n'
            'Кнопка "Просмотр отчетов" открывает доступ к результатам '
            'анализа и сформированным отчетам.'
        )
        buttons_description.setStyleSheet('''
            QLabel {
                font-size: 12px;
                font-family: Roboto;
                color: #878787;
                line-height: 100%;
                letter-spacing: 0%;
            }
        ''')
        buttons_description.setWordWrap(True)
        left_layout.addWidget(buttons_description)

        left_layout.addStretch()

        # Правый макет
        right_widget = QtWidgets.QWidget()
        right_widget.setStyleSheet('''
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #C9D6FF,
                    stop:0.6572 #E2E2E2);
            }
        ''')
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(30, 30, 30, 30)  # Отступы для содержимого правой части
        main_layout.addWidget(right_widget, stretch=1)

        # Контейнер для фото и кнопок с вертикальным расположением
        content_layout = QtWidgets.QVBoxLayout()
        content_layout.setSpacing(30)  # Увеличиваем расстояние между элементами
        right_layout.addLayout(content_layout)
        right_layout.addStretch()  # Прижимаем содержимое к верху

        # Добавляем отступ перед фото
        content_layout.addSpacing(20)  # Уменьшили с 40 до 20

        # Круглое фото
        icon_label = QtWidgets.QLabel()
        icon_size = 250  # Увеличили размер с 200 до 250
        icon_label.setFixedSize(icon_size, icon_size)
        icon_label.setStyleSheet('''
            QLabel {
                background: transparent;
            }
        ''')
        
        # Загрузка и установка круглого фото
        pixmap = QtGui.QPixmap(r'C:\Users\User\PycharmProjects\pythonProject\images\sours/фото.jfif')
        # Увеличиваем размер изображения перед масштабированием
        scaled_pixmap = pixmap.scaled(icon_size + 50, icon_size + 50, 
                                    QtCore.Qt.KeepAspectRatioByExpanding, 
                                    QtCore.Qt.SmoothTransformation)
        
        # Центрируем изображение
        x = (scaled_pixmap.width() - icon_size) // 2
        y = (scaled_pixmap.height() - icon_size) // 2
        scaled_pixmap = scaled_pixmap.copy(x, y, icon_size, icon_size)
        
        # Создаем маску для круглого изображения
        mask = QtGui.QPixmap(icon_size, icon_size)
        mask.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(mask)
        painter.setBrush(QtCore.Qt.white)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(0, 0, icon_size, icon_size)
        painter.end()
        
        # Применяем маску к изображению
        rounded_pixmap = QtGui.QPixmap(scaled_pixmap.size())
        rounded_pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(rounded_pixmap)
        painter.setClipRegion(QtGui.QRegion(mask.mask()))
        painter.drawPixmap(0, 0, scaled_pixmap)
        painter.end()
        
        icon_label.setPixmap(rounded_pixmap)
        content_layout.addWidget(icon_label, alignment=QtCore.Qt.AlignCenter)
        
        # Добавляем дополнительное пространство перед кнопками
        content_layout.addSpacing(50)  # Увеличиваем расстояние между фото и кнопками

        # Макет для кнопок с горизонтальным расположением
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(15)
        content_layout.addLayout(button_layout)

        # Кнопка загрузки изображений
        self.upload_button = QtWidgets.QPushButton('Загрузить фото')
        self.upload_button.setFixedSize(160, 35)  # Уменьшенный размер кнопки
        self.upload_button.setStyleSheet('''
            QPushButton {
                background: #3DC3BC;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 17px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #03204f;
            }
        ''')
        button_layout.addWidget(self.upload_button, alignment=QtCore.Qt.AlignCenter)

        # Кнопка просмотра отчета
        self.view_report_button = QtWidgets.QPushButton('Просмотр отчетов')
        self.view_report_button.setFixedSize(160, 35)  # Уменьшенный размер кнопки
        self.view_report_button.setStyleSheet('''
            QPushButton {
                background: #8AA4BE;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 17px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #03204f;
            }
        ''')
        button_layout.addWidget(self.view_report_button, alignment=QtCore.Qt.AlignCenter)

        # Подключение сигналов к слотам
        self.upload_button.clicked.connect(self.upload_images)
        self.view_report_button.clicked.connect(self.view_report)

    def upload_images(self):
        # Открываем диалог выбора файлов
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)")
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            # Создаем и показываем окно выбора изображений
            self.selection_window = ImageSelectionWindow(selected_files)
            self.selection_window.show()

    def view_report(self):
        self.report_window = ReportViewerWindow()
        self.report_window.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    app_icon = QIcon('C:/Users/User/PycharmProjects/pythonProject/images/sours/icon.png')
    app.setWindowIcon(app_icon)
    window = MainWindow()
    window.show()
    app.exec_() 