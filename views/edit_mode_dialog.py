from PyQt5 import QtWidgets, QtGui, QtCore
from single_edit_window import SingleEditWindow
from multi_edit_window import MultiEditWindow

class EditModeDialog(QtWidgets.QDialog):
    """Диалог для выбора режима редактирования"""
    
    def __init__(self, analysis_dir):
        super().__init__()
        self.analysis_dir = analysis_dir
        self.single_edit_window = None
        self.multi_edit_window = None
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle('Выбор режима редактирования')
        self.setMinimumWidth(400)
        
        # Устанавливаем белый фон для окна
        self.setStyleSheet('''
            QDialog {
                background: white;
                border-radius: 15px;
            }
        ''')
        
        # Устанавливаем иконку окна
        icon = QtGui.QIcon('C:/Users/User/PycharmProjects/pythonProject/images/sours/icon.png')
        self.setWindowIcon(icon)
        
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(5)  # Уменьшаем расстояние между всеми элементами до 5px
        self.setLayout(layout)
        
        # Заголовок
        title_label = QtWidgets.QLabel('Анализ завершен')
        title_label.setStyleSheet('''
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #464646;
                padding: 3px;
                margin-bottom: 2px;
            }
        ''')
        layout.addWidget(title_label)
        
        # Информация о завершении
        completion_label = QtWidgets.QLabel('Анализ успешно завершен. Желаете отредактировать результаты?')
        completion_label.setStyleSheet('''
            QLabel {
                color: #5D5D5D;
                padding: 3px;
                font-size: 14px;
                margin-bottom: 2px;
            }
        ''')
        completion_label.setWordWrap(True)
        layout.addWidget(completion_label)
        
        # Информация о режимах
        info_label = QtWidgets.QLabel(
            'Одиночное редактирование позволяет просматривать и удалять '
            'отдельные найденные области.\n\n'
            'Множественное редактирование позволяет удалять и добавлять '
            'области на изображении результатов.'
        )
        info_label.setStyleSheet('''
            QLabel {
                color: #878787;
                padding: 3px;
                font-size: 13px;
                margin-bottom: 5px;
            }
        ''')
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Кнопки редактирования
        button_layout = QtWidgets.QHBoxLayout()
        
        button_style = '''
            QPushButton {
                background-color: #8AA4BE;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 15px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #7B93AB;
            }
        '''
        
        self.single_edit_button = QtWidgets.QPushButton('Одиночное редактирование')
        self.single_edit_button.setStyleSheet(button_style)
        self.single_edit_button.clicked.connect(self.start_single_edit)
        button_layout.addWidget(self.single_edit_button)
        
        self.multi_edit_button = QtWidgets.QPushButton('Множественное редактирование')
        self.multi_edit_button.setStyleSheet(button_style)
        self.multi_edit_button.clicked.connect(self.start_multi_edit)
        button_layout.addWidget(self.multi_edit_button)
        
        layout.addLayout(button_layout)
        
        # Кнопка пропуска
        self.skip_button = QtWidgets.QPushButton('Пропустить редактирование')
        self.skip_button.setStyleSheet('''
            QPushButton {
                background-color: #FF5959;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 15px;
                font-size: 14px;
                min-width: 120px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #E65252;
            }
        ''')
        self.skip_button.clicked.connect(self.reject)
        layout.addWidget(self.skip_button)
        
        # Уменьшаем отступы для всего окна
        layout.setContentsMargins(15, 10, 15, 10)  # Уменьшили вертикальные отступы
    
    def start_single_edit(self):
        """Запускает режим одиночного редактирования"""
        try:
            self.accept()
            self.single_edit_window = SingleEditWindow(self.analysis_dir)
            self.single_edit_window.show()
            print(f"Открыто окно одиночного редактирования для {self.analysis_dir}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                'Ошибка',
                f'Не удалось открыть окно редактирования: {str(e)}'
            )
            print(f"Ошибка: {str(e)}")
    
    def start_multi_edit(self):
        """Запускает режим множественного редактирования"""
        try:
            self.accept()
            self.multi_edit_window = MultiEditWindow(self.analysis_dir)
            self.multi_edit_window.show()
            print(f"Открыто окно множественного редактирования для {self.analysis_dir}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                'Ошибка',
                f'Не удалось открыть окно множественного редактирования: {str(e)}'
            )
            print(f"Ошибка: {str(e)}") 