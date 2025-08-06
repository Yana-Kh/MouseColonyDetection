from PyQt5 import QtWidgets, QtGui, QtCore
import os
from datetime import datetime, timedelta
from database import Database
import sqlite3
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

class ReportViewerWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.setup_ui()
        self.load_reports()

    def setup_ui(self):
        self.setWindowTitle('Просмотр отчетов')
        self.setGeometry(200, 200, 800, 600)

        # Основной макет
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)
        
        # Устанавливаем белый фон и закругленные углы для всего окна
        self.setStyleSheet('''
            QWidget {
                background: white;
                border-radius: 30px;
            }
            QGroupBox {
                border: 1px solid #e0e0e0;
                border-radius: 15px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        ''')

        # Верхняя панель с выбором периода
        period_group = QtWidgets.QGroupBox('Выбор периода')
        period_layout = QtWidgets.QHBoxLayout()
        period_group.setLayout(period_layout)
        period_group.setStyleSheet('''
            QGroupBox {
                background: #F5F5F5;
                border: none;
                border-radius: 15px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                color: #464646;
            }
            QLabel {
                color: #464646;
            }
            QDateEdit {
                background: white;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                padding: 4px;
            }
        ''')

        # Метки и поля для выбора дат
        period_layout.addWidget(QtWidgets.QLabel('С:'))
        self.date_from = QtWidgets.QDateEdit(calendarPopup=True)
        self.date_from.setDate(QtCore.QDate.currentDate().addDays(-30))
        period_layout.addWidget(self.date_from)

        period_layout.addWidget(QtWidgets.QLabel('По:'))
        self.date_to = QtWidgets.QDateEdit(calendarPopup=True)
        self.date_to.setDate(QtCore.QDate.currentDate())
        period_layout.addWidget(self.date_to)

        # Кнопка применения фильтра по дате (уменьшенная)
        self.apply_filter_button = QtWidgets.QPushButton('Применить')
        self.apply_filter_button.setStyleSheet('''
            QPushButton {
                background-color: #6EC96E;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 12px;
                font-size: 12px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #5DB85D;
            }
        ''')
        self.apply_filter_button.clicked.connect(self.apply_date_filter)
        period_layout.addWidget(self.apply_filter_button)

        main_layout.addWidget(period_group)

        # Список анализов
        reports_group = QtWidgets.QGroupBox('Доступные анализы')
        reports_layout = QtWidgets.QVBoxLayout()
        reports_group.setLayout(reports_layout)

        self.reports_list = QtWidgets.QListWidget()
        self.reports_list.itemDoubleClicked.connect(self.open_report)
        reports_layout.addWidget(self.reports_list)

        main_layout.addWidget(reports_group)

        # Кнопки управления
        button_layout = QtWidgets.QHBoxLayout()
        
        # Обновляем стиль для кнопок управления
        open_button_style = '''
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
        '''

        close_button_style = '''
            QPushButton {
                background-color: #FF5959;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 15px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #E65252;
            }
        '''

        self.open_button = QtWidgets.QPushButton('Открыть анализ')
        self.open_button.setStyleSheet(open_button_style)
        self.open_button.clicked.connect(self.open_selected_report)
        button_layout.addWidget(self.open_button)

        self.close_button = QtWidgets.QPushButton('Закрыть')
        self.close_button.setStyleSheet(close_button_style)
        self.close_button.clicked.connect(self.hide)
        button_layout.addWidget(self.close_button)

        # Обновляем стиль кнопок экспорта с более круглыми углами
        export_layout = QtWidgets.QHBoxLayout()
        
        export_button_style = '''
            QPushButton {
                background-color: #8AA4BE;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 15px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #7B93AB;
            }
        '''
        
        self.export_pdf_button = QtWidgets.QPushButton('Экспорт в PDF')
        self.export_pdf_button.clicked.connect(self.export_to_pdf)
        self.export_pdf_button.setStyleSheet(export_button_style)
        export_layout.addWidget(self.export_pdf_button)
        
        self.export_excel_button = QtWidgets.QPushButton('Экспорт в Excel')
        self.export_excel_button.clicked.connect(self.export_to_excel)
        self.export_excel_button.setStyleSheet(export_button_style)
        export_layout.addWidget(self.export_excel_button)

        main_layout.addLayout(button_layout)
        main_layout.addLayout(export_layout)

    def load_reports(self):
        """Загрузка списка анализов из базы данных"""
        self.reports_list.clear()
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Получаем все анализы с датой создания
        cursor.execute('''
            SELECT analysis_id, created_at, 
                   (SELECT COUNT(*) FROM Image WHERE Image.analysis_id = Analysis.analysis_id) as image_count,
                   (SELECT COUNT(*) FROM Colony WHERE Colony.image_id IN 
                    (SELECT image_id FROM Image WHERE Image.analysis_id = Analysis.analysis_id)) as colony_count
            FROM Analysis
            ORDER BY created_at DESC
        ''')
        
        analyses = cursor.fetchall()
        conn.close()

        for analysis_id, created_at, image_count, colony_count in analyses:
            created_date = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            item = QtWidgets.QListWidgetItem(
                f"Анализ от {created_date.strftime('%d.%m.%Y %H:%M')} "
                f"(Изображений: {image_count}, Колоний: {colony_count})"
            )
            item.setData(QtCore.Qt.UserRole, analysis_id)
            self.reports_list.addItem(item)

    def apply_date_filter(self):
        """Применение фильтра по датам"""
        date_from = self.date_from.date().toPyDate()
        date_to = self.date_to.date().toPyDate()

        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT analysis_id, created_at,
                   (SELECT COUNT(*) FROM Image WHERE Image.analysis_id = Analysis.analysis_id) as image_count,
                   (SELECT COUNT(*) FROM Colony WHERE Colony.image_id IN 
                    (SELECT image_id FROM Image WHERE Image.analysis_id = Analysis.analysis_id)) as colony_count
            FROM Analysis
            WHERE date(created_at) BETWEEN ? AND ?
            ORDER BY created_at DESC
        ''', (date_from.strftime('%Y-%m-%d'), date_to.strftime('%Y-%m-%d')))
        
        analyses = cursor.fetchall()
        conn.close()

        self.reports_list.clear()
        for analysis_id, created_at, image_count, colony_count in analyses:
            created_date = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            item = QtWidgets.QListWidgetItem(
                f"Анализ от {created_date.strftime('%d.%m.%Y %H:%M')} "
                f"(Изображений: {image_count}, Колоний: {colony_count})"
            )
            item.setData(QtCore.Qt.UserRole, analysis_id)
            self.reports_list.addItem(item)

    def open_selected_report(self):
        """Открытие выбранного анализа"""
        current_item = self.reports_list.currentItem()
        if current_item:
            self.open_report(current_item)

    def open_report(self, item):
        """Открытие анализа для просмотра"""
        analysis_id = item.data(QtCore.Qt.UserRole)
        try:
            # Создаем новое окно для отображения данных
            report_display = ReportDisplayDialog(analysis_id, self.db, self)
            report_display.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                None,
                'Ошибка',
                f'Не удалось открыть анализ: {str(e)}'
            )

    def export_to_pdf(self):
        """Экспортирует текущий отчет в PDF"""
        try:
            current_item = self.reports_list.currentItem()
            if not current_item:
                QtWidgets.QMessageBox.warning(None, 'Предупреждение', 'Выберите анализ для экспорта')
                return
            
            analysis_id = current_item.data(QtCore.Qt.UserRole)
            
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Сохранить PDF',
                f'Анализ_{analysis_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                'PDF files (*.pdf)'
            )
            
            if not file_path:
                return
            
            # Создаем PDF с поддержкой русского языка
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # Регистрируем шрифт Arial для поддержки русского языка
            pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
            
            doc = SimpleDocTemplate(
                file_path,
                pagesize=landscape(letter),
                rightMargin=20,
                leftMargin=20,
                topMargin=20,
                bottomMargin=20
            )
            
            elements = []
            styles = getSampleStyleSheet()
            
            # Создаем стиль с русским шрифтом
            styles['Normal'].fontName = 'Arial'
            styles['Heading1'].fontName = 'Arial'
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Получаем информацию об анализе
                cursor.execute('''
                    SELECT created_at,
                           (SELECT COUNT(*) FROM Image WHERE analysis_id = ?) as image_count,
                           (SELECT COUNT(*) FROM Colony 
                            WHERE image_id IN (SELECT image_id FROM Image WHERE analysis_id = ?)) as colony_count
                    FROM Analysis 
                    WHERE analysis_id = ?
                ''', (analysis_id, analysis_id, analysis_id))
                
                analysis_info = cursor.fetchone()
                
                # Добавляем заголовок и общую информацию
                created_at = datetime.strptime(analysis_info[0], '%Y-%m-%d %H:%M:%S') + timedelta(hours=3)
                elements.append(Paragraph(f"Отчет по анализу #{analysis_id}", styles['Heading1']))
                elements.append(Paragraph(f"Дата создания: {created_at.strftime('%d.%m.%Y %H:%M')}", styles['Normal']))
                elements.append(Paragraph(f"Всего изображений: {analysis_info[1]}", styles['Normal']))
                elements.append(Paragraph(f"Всего колоний: {analysis_info[2]}", styles['Normal']))
                elements.append(Paragraph("<br/><br/>", styles['Normal']))
                
                # Получаем данные о колониях
                cursor.execute('''
                    SELECT 
                        i.file_path,
                        i.date,
                        i.latitude as img_lat,
                        i.longitude as img_lon,
                        i.altitude,
                        c.latitude as colony_lat,
                        c.longitude as colony_lon,
                        c.confidence,
                        c.verified,
                        c.x,
                        c.y,
                        c.window_path
                    FROM Colony c
                    JOIN Image i ON c.image_id = i.image_id
                    WHERE i.analysis_id = ?
                    ORDER BY i.file_path, c.confidence DESC
                ''', (analysis_id,))
                
                colonies_data = cursor.fetchall()
                
                # Создаем таблицу с данными
                table_data = [[
                    'Файл', 
                    'Дата', 
                    'Координаты\nизображения',
                    'Высота',
                    'Координаты\nколонии', 
                    'Уверенность',
                    'Проверено',
                    'Координаты\nокна (X, Y)'
                ]]
                
                for colony in colonies_data:
                    # Сокращаем путь к файлу, оставляя только последние две папки и имя файла
                    path_parts = colony[0].split(os.sep)
                    if len(path_parts) > 2:
                        filename = os.path.join(*path_parts[-3:])
                    else:
                        filename = colony[0]
                    
                    # Форматируем дату более компактно
                    date_str = str(colony[1]) if colony[1] else ""
                    if len(date_str) > 10:
                        try:
                            # Преобразуем строку в datetime и добавляем 3 часа для корректировки часового пояса
                            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') + timedelta(hours=3)
                            date_str = dt.strftime('%d.%m.%Y\n%H:%M')
                        except:
                            date_str = date_str.replace(" ", "\n")
                    
                    table_data.append([
                        filename,
                        date_str,
                        f"({colony[2]:.6f},\n{colony[3]:.6f})",
                        f"{colony[4]:.1f} м",
                        f"({colony[5]:.6f},\n{colony[6]:.6f})",
                        f"{colony[7]:.3f}",
                        'Да' if colony[8] else 'Нет',
                        f"({colony[9]},\n{colony[10]})"
                    ])
                
                # Настраиваем ширину столбцов (в точках)
                col_widths = [250, 80, 90, 45, 90, 50, 45, 60]  # Увеличили ширину первого столбца до 250
                table = Table(table_data, colWidths=col_widths, repeatRows=1)
                
                # Обновляем стиль таблицы
                table.setStyle(TableStyle([
                    # Стиль заголовков
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Arial'),
                    ('FONTSIZE', (0, 0), (-1, 0), 7),  # Уменьшили размер шрифта заголовков до 7
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),  # Уменьшили отступ снизу
                    
                    # Стиль данных
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Arial'),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),  # Оставили тот же размер для данных
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('LEFTPADDING', (0, 0), (-1, -1), 2),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                    
                    # Разрешаем перенос текста во всех ячейках
                    ('WORDWRAP', (0, 0), (-1, -1), True),
                    
                    # Чередующиеся цвета строк
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
                    
                    # Выравнивание для конкретных столбцов
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Имена файлов по левому краю
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),  # Остальные данные по центру
                    
                    # Уменьшаем межстрочный интервал
                    ('LEADING', (0, 0), (-1, -1), 7),  # Уменьшили межстрочный интервал
                ]))
                
                elements.append(table)
            
            # Создаем PDF
            doc.build(elements)
        
            QtWidgets.QMessageBox.information(
                None,
                'Успех',
                f'Отчет сохранен в файл:\n{file_path}'
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                'Ошибка',
                f'Не удалось создать PDF:\n{str(e)}'
            )
    
    def export_to_excel(self):
        """Экспортирует текущий отчет в Excel"""
        try:
            current_item = self.reports_list.currentItem()
            if not current_item:
                QtWidgets.QMessageBox.warning(None, 'Предупреждение', 'Выберите анализ для экспорта')
                return
            
            analysis_id = current_item.data(QtCore.Qt.UserRole)
            
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Сохранить Excel',
                f'Анализ_{analysis_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                'Excel files (*.xlsx)'
            )
            
            if not file_path:
                return
            
            with sqlite3.connect(self.db.db_path) as conn:
                # Получаем данные о колониях
                df = pd.read_sql_query('''
                    SELECT 
                        i.file_path as "Путь к файлу",
                        i.date as "Дата",
                        i.latitude as "Широта изображения",
                        i.longitude as "Долгота изображения",
                        i.altitude as "Высота (м)",
                        c.latitude as "Широта колонии",
                        c.longitude as "Долгота колонии",
                        c.confidence as "Уверенность",
                        c.verified as "Проверено",
                        c.x as "X",
                        c.y as "Y",
                        c.window_path as "Путь к окну"
                    FROM Colony c
                    JOIN Image i ON c.image_id = i.image_id
                    WHERE i.analysis_id = ?
                    ORDER BY i.file_path, c.confidence DESC
                ''', conn, params=(analysis_id,))
                
                # Форматируем числовые колонки
                df["Уверенность"] = df["Уверенность"].apply(lambda x: f"{x:.3f}")
                df["Проверено"] = df["Проверено"].apply(lambda x: "Да" if x else "Нет")
                for col in ["Широта изображения", "Долгота изображения", "Широта колонии", "Долгота колонии"]:
                    df[col] = df[col].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else "")
                df["Высота (м)"] = df["Высота (м)"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
                
                # После получения DataFrame добавим корректировку времени:
                df["Дата"] = pd.to_datetime(df["Дата"]).apply(lambda x: (x + timedelta(hours=3)).strftime('%d.%m.%Y %H:%M') if pd.notnull(x) else "")
                
                # Создаем Excel writer
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    # Записываем данные
                    df.to_excel(writer, sheet_name='Отчет', index=False)
                    
                    # Получаем workbook и worksheet
                    workbook = writer.book
                    worksheet = writer.sheets['Отчет']
                    
                    # Форматирование заголовков
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'vcenter',
                        'align': 'center',
                        'bg_color': '#4F81BD',
                        'font_color': 'white',
                        'border': 1
                    })
                    
                    # Форматирование данных
                    data_format = workbook.add_format({
                        'align': 'left',
                        'valign': 'vcenter',
                        'border': 1
                    })
                    
                    # Применяем форматирование
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                        # Устанавливаем ширину колонок в зависимости от содержимого
                        max_length = max(
                            df[value].astype(str).apply(len).max(),
                            len(value)
                        )
                        worksheet.set_column(col_num, col_num, max_length + 2)
                    
                    # Применяем форматирование к данным
                    for row in range(1, len(df) + 1):
                        for col in range(len(df.columns)):
                            worksheet.write(row, col, df.iloc[row-1, col], data_format)
            
            QtWidgets.QMessageBox.information(
                None,
                'Успех',
                f'Отчет сохранен в файл:\n{file_path}'
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                'Ошибка',
                f'Не удалось создать Excel файл:\n{str(e)}'
            )

class ReportDisplayDialog(QtWidgets.QDialog):
    def __init__(self, analysis_id, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.analysis_id = analysis_id
        self.setWindowTitle('Просмотр анализа')
        self.setGeometry(250, 250, 1000, 600)
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Создаем таблицу
        self.table = QtWidgets.QTableWidget()
        layout.addWidget(self.table)

        # Кнопка закрытия
        close_button = QtWidgets.QPushButton('Закрыть')
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button) 

    def load_data(self):
        """Загружает данные анализа из базы данных"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        # Получаем все колонии для данного анализа с информацией об изображениях
        cursor.execute('''
            SELECT 
                Image.file_path,
                Image.date,
                Image.latitude as img_lat,
                Image.longitude as img_lon,
                Colony.latitude as colony_lat,
                Colony.longitude as colony_lon,
                Colony.confidence,
                Colony.verified,
                Colony.x,
                Colony.y,
                Colony.window_path
            FROM Colony
            JOIN Image ON Colony.image_id = Image.image_id
            WHERE Image.analysis_id = ?
            ORDER BY Image.file_path, Colony.confidence DESC
        ''', (self.analysis_id,))

        colonies = cursor.fetchall()
        conn.close()

        # Настраиваем таблицу
        headers = [
            'Изображение', 'Дата', 
            'Широта изобр.', 'Долгота изобр.',
            'Широта колонии', 'Долгота колонии',
            'Уверенность', 'Проверено',
            'X', 'Y', 'Путь к окну'
        ]
        
        self.table.setColumnCount(len(headers))
        self.table.setRowCount(len(colonies))
        self.table.setHorizontalHeaderLabels(headers)

        # Заполняем таблицу данными
        for row, colony in enumerate(colonies):
            for col, value in enumerate(colony):
                if col == 7:  # Колонка "Проверено"
                    value = "Да" if value else "Нет"
                elif col in [2, 3, 4, 5]:  # Координаты
                    value = f"{value:.6f}" if value else ""
                elif col == 6:  # Уверенность
                    value = f"{value:.3f}" if value else ""
                
                item = QtWidgets.QTableWidgetItem(str(value))
                self.table.setItem(row, col, item)

        # Растягиваем столбцы по содержимому
        self.table.resizeColumnsToContents() 