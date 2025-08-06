import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_path='analysis.db'):
        self.db_path = db_path
        self.create_tables()
    
    def create_tables(self):
        """Создает необходимые таблицы в базе данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создаем таблицу анализов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Analysis (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Создаем таблицу изображений
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Image (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TIMESTAMP,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            file_path TEXT NOT NULL,
            analysis_id INTEGER,
            FOREIGN KEY (analysis_id) REFERENCES Analysis(analysis_id)
        )
        ''')
        
        # Создаем таблицу колоний с дополнительными полями
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Colony (
            colony_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            latitude REAL,
            longitude REAL,
            confidence REAL,
            verified BOOLEAN DEFAULT FALSE,
            x INTEGER,           -- Координата X на изображении
            y INTEGER,           -- Координата Y на изображении
            window_path TEXT,    -- Путь к вырезанному окну
            FOREIGN KEY (image_id) REFERENCES Image(image_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_analysis(self):
        """Создает новую запись анализа и возвращает его ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO Analysis DEFAULT VALUES')
        analysis_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return analysis_id
    
    def add_image(self, file_path, date, latitude, longitude, altitude, analysis_id):
        """Добавляет информацию об изображении"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO Image (file_path, date, latitude, longitude, altitude, analysis_id)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (file_path, date, latitude, longitude, altitude, analysis_id))
        
        image_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return image_id
    
    def add_colony(self, image_id, latitude, longitude, confidence, x, y, window_path, verified=False):
        """Добавляет информацию о найденной колонии"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO Colony (
            image_id, latitude, longitude, confidence, 
            x, y, window_path, verified
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (image_id, latitude, longitude, confidence, x, y, window_path, verified))
        
        colony_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return colony_id
    
    def update_colony_verification(self, colony_id, verified):
        """Обновляет статус проверки колонии"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE Colony
        SET verified = ?
        WHERE colony_id = ?
        ''', (verified, colony_id))
        
        conn.commit()
        conn.close()
    
    def get_analysis_images(self, analysis_id):
        """Получает все изображения для конкретного анализа"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT image_id, file_path, date, latitude, longitude, altitude
        FROM Image
        WHERE analysis_id = ?
        ''', (analysis_id,))
        
        images = cursor.fetchall()
        
        conn.close()
        
        return images
    
    def get_image_colonies(self, image_id):
        """Получает все колонии для конкретного изображения"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT colony_id, latitude, longitude, confidence, verified
        FROM Colony
        WHERE image_id = ?
        ''', (image_id,))
        
        colonies = cursor.fetchall()
        
        conn.close()
        
        return colonies
    
    def mark_colonies_verified(self, image_id):
        """Отмечает все колонии изображения как проверенные"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE Colony
        SET verified = TRUE
        WHERE image_id = ?
        ''', (image_id,))
        
        conn.commit()
        conn.close() 