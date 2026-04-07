# theme_system.py - Font desteği ve #cvSummary stil desteği ile güncel tema sistemi

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow

class ThemeManager:
    """Tema yönetimi için ayrı sınıf"""

    @staticmethod
    def get_dark_theme_stylesheet(font_family="Segoe UI", font_size=10):
        """Koyu tema stil sayfası - font parametreleri ile"""
        return f"""
            QMainWindow {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QWidget {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QFrame {{
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 5px;
                font-family: '{font_family}';
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
                background-color: #363636;
                font-family: '{font_family}';
                font-size: {font_size + 1}pt;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #ffffff;
                background-color: #363636;
                font-family: '{font_family}';
                font-weight: bold;
            }}
            QPushButton {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QPushButton:hover {{
                background-color: #45a049;
            }}
            QPushButton:pressed {{
                background-color: #367c39;
            }}
            QLineEdit, QTextEdit {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            /* Özet alanı için özel stil (koyu tema) */
            #cvSummary {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #4CAF50;
                selection-color: #ffffff;
            }}

            QProgressBar {{
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #777777;
                border-radius: 5px;
                text-align: center;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QProgressBar::chunk {{
                background-color: #4CAF50;
                border-radius: 5px;
            }}
            QTreeWidget {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                alternate-background-color: #363636;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QHeaderView::section {{
                background-color: #444444;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555555;
                font-family: '{font_family}';
                font-size: {font_size}pt;
                font-weight: bold;
            }}
            QLabel {{
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QCheckBox, QRadioButton {{
                color: #ffffff;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QComboBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 1px 18px 1px 3px;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 3px;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
        """

    @staticmethod
    def get_light_theme_stylesheet(font_family="Segoe UI", font_size=10):
        """Açık tema stil sayfası - font parametreleri ile"""
        return f"""
            QMainWindow {{
                background-color: #f0f0f0;
                color: #333333;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QWidget {{
                background-color: #f0f0f0;
                color: #333333;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QFrame {{
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 5px;
                font-family: '{font_family}';
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
                background-color: #f8f8f8;
                font-family: '{font_family}';
                font-size: {font_size + 1}pt;
            }}
            QPushButton {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QLineEdit, QTextEdit {{
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            /* Özet alanı için özel stil (açık tema) */
            #cvSummary {{
                background-color: #f8f9fa;
                color: #212529;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
            }}
            QTreeWidget {{
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 5px;
                alternate-background-color: #f5f5f5;
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
            QLabel {{
                font-family: '{font_family}';
                font-size: {font_size}pt;
            }}
        """

    @staticmethod
    def apply_theme(main_window: QMainWindow, settings: dict):
        """
        Ayarlara göre temayı ana pencereye uygular.
        Font ayarlarını da dikkate alır.
        """
        font_obj = settings.get('font', QFont("Segoe UI", 10))
        font_family = font_obj.family()
        font_size = font_obj.pointSize()

        if settings.get('current_theme') == 'dark':
            stylesheet = ThemeManager.get_dark_theme_stylesheet(font_family, font_size)
        else:
            stylesheet = ThemeManager.get_light_theme_stylesheet(font_family, font_size)

        main_window.setStyleSheet(stylesheet)
