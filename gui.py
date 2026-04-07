# gui.py - Güncellenmiş sürüm (özet varsayılan kapalı, koyu temaya uyum, sağ üst "Prompt'a göre Analiz" butonu, prompt ekranında Aç/Resmi Gör/Özet)
from typing import Dict, List, Any, Optional
import os
import sys
import json
import csv
import io
import re

# ============================================================================
# QT PLATFORM PLUGIN HATASI ÇÖZÜMÜ - Bu bölüm PyQt5 import'larından ÖNCE olmalı
# ============================================================================
if hasattr(sys, 'frozen'):
    os.environ['QT_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt5', 'plugins')
else:
    # Sanal ortam için plugin yolunu ayarla
    venv_path = os.path.dirname(sys.executable)
    plugin_path = os.path.join(venv_path, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
    
    # Alternatif yolları da kontrol et
    if not os.path.exists(plugin_path):
        # Python site-packages doğrudan erişim
        import site
        site_packages = site.getsitepackages()
        for sp in site_packages:
            potential_path = os.path.join(sp, 'PyQt5', 'Qt5', 'plugins')
            if os.path.exists(potential_path):
                plugin_path = potential_path
                break
    
    if os.path.exists(plugin_path):
        os.environ['QT_PLUGIN_PATH'] = plugin_path
        print(f"Qt plugin yolu ayarlandı: {plugin_path}")
    else:
        print(f"UYARI: Qt plugin yolu bulunamadı: {plugin_path}")
# ============================================================================

import cv2
import numpy as np
from PIL import Image
import fitz

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QTextEdit, QFileDialog,
                            QProgressBar, QMessageBox, QTreeWidget, QTreeWidgetItem,
                            QSplitter, QGroupBox, QFrame, QLineEdit, QAction,
                            QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
                            QSlider, QApplication, QColorDialog, QFontDialog, QInputDialog,
                            QRadioButton, QComboBox, QHeaderView, QSizePolicy, QStackedWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QDateTime
from PyQt5.QtGui import QFont, QIcon, QDesktopServices, QColor, QPixmap

# ... Kodun geri kalanı aynı kalacak

# Yerel modüller
from analyzer import CVAnalyzer
import theme_system
from cv_analyzer import model, analyze_cv_folder, clean_text


# -----------------------------
# Resim işleme yardımcıları
# -----------------------------
def is_face_image(image_bytes):
    """Verilen resmin yüz içerip içermediğini kontrol eder."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return len(faces) > 0

    except Exception as e:
        print(f"Yüz tespiti hatası: {e}")
        return False


def filter_profile_images(images):
    """Resimler listesinden sadece yüz içerenleri filtreler ve en iyi adayı döndürür."""
    profile_images = []

    for img_data in images:
        try:
            img = Image.open(io.BytesIO(img_data))
            width, height = img.size

            # Boyut filtreleme
            if width < 64 or height < 64 or width > 1024 or height > 1024:
                continue

            # En-boy oranı kontrolü
            aspect_ratio = width / height
            if aspect_ratio < 0.33 or aspect_ratio > 3.0:
                continue

            if is_face_image(img_data):
                profile_images.append({
                    'data': img_data, 'width': width, 'height': height, 'size': len(img_data)
                })

        except Exception as e:
            print(f"Resim filtreleme hatası: {e}")
            continue

    if profile_images:
        # En büyük/kaliteli adayı seç
        profile_images.sort(key=lambda x: x['width'] * x['height'] * x['size'], reverse=True)
        return profile_images[0]['data']

    return None


def extract_profile_picture_from_pdf_enhanced(pdf_path):
    """PDF'den profil resmi çıkarır (ilk 3 sayfayı tarar, yüz içeren en iyi resmi seçer)."""
    try:
        doc = fitz.open(pdf_path)
        all_images = []

        max_pages = min(3, len(doc))
        for page_num in range(max_pages):
            page = doc.load_page(page_num)
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                    else:
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("png")
                        pix1 = None
                    pix = None
                    all_images.append(img_data)

                except Exception as e:
                    print(f"Resim çıkarma hatası (sayfa {page_num}, resim {img_index}): {e}")
                    continue

        doc.close()
        return filter_profile_images(all_images)

    except Exception as e:
        print(f"PDF işleme hatası: {e}")
        return None


# -----------------------------
# Ağaç öğesi: sayısal sıralama
# -----------------------------
class NumericTreeWidgetItem(QTreeWidgetItem):
    """Sayısal sütunları doğru sıralamak için özel TreeWidgetItem."""
    def __init__(self, parent=None):
        super().__init__(parent)

    def __lt__(self, other):
        column = self.treeWidget().sortColumn()

        if column == 1:
            try:
                val1 = self.data(column, Qt.UserRole)
                val2 = other.data(column, Qt.UserRole)
                if val1 is not None and val2 is not None:
                    return float(val1) < float(val2)

                text1 = self.text(column).replace('%', '').replace(',', '.')
                text2 = other.text(column).replace('%', '').replace(',', '.')
                return float(text1) < float(text2)
            except (ValueError, AttributeError):
                pass
        return self.text(column) < other.text(column)


# -----------------------------
# Ayarlar (Gelişmiş)
# -----------------------------
class PreferencesDialog(QDialog):
    """Gelişmiş ayarlar dialogu."""
    def __init__(self, settings: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.setGeometry(200, 200, 500, 400)

        self.settings = settings.copy()
        self.selected_font = self.settings.get('font', QApplication.font())

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self._create_analysis_settings_group()
        self._create_text_processing_settings_group()
        self._create_ui_settings_group()
        self._create_buttons()

        self._load_settings_to_ui()
        self._apply_dialog_styles()

    def _apply_dialog_styles(self):
        """Dialog için özel stil sayfası uygula."""
        dialog_styles = """
            QPushButton {
                min-width: 80px; min-height: 30px; padding: 6px 12px;
                border-radius: 4px; font-weight: 500;
            }
            QPushButton[class="success"] { 
                background-color: #28a745; color: white; border: 1px solid #1e7e34; 
            }
            QPushButton[class="success"]:hover { 
                background-color: #34ce57; border: 2px solid #1e7e34; 
            }
            QPushButton[class="secondary"] { 
                background-color: #6c757d; color: white; border: 1px solid #545b62; 
            }
            QPushButton[class="secondary"]:hover { 
                background-color: #7c838a; border: 2px solid #545b62; 
            }
        """
        self.setStyleSheet(dialog_styles)

    def _create_analysis_settings_group(self):
        group_box = QGroupBox("Analiz Ayarları")
        form_layout = QFormLayout()

        self.min_similarity_spinbox = QSpinBox()
        self.min_similarity_spinbox.setRange(0, 100)
        form_layout.addRow("Minimum Benzerlik Eşiği (%):", self.min_similarity_spinbox)

        # Bölüm ağırlıkları için basit yapı
        section_weights_group = QGroupBox("Bölüm AĞIRLIK")
        self.section_weights_layout = QFormLayout()
        self.section_sliders = {}
        self.section_labels = {}

        default_section_weights = {
            'personal_info': 10, 'education': 20, 'experience': 40,
            'skills': 20, 'projects': 5, 'certifications': 5
        }

        for section, default_weight in default_section_weights.items():
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(default_weight)
            slider.valueChanged.connect(self._update_section_weights_display)
            self.section_sliders[section] = slider

            label = QLabel(f"{section.replace('_', ' ').title()}: {slider.value()}%")
            self.section_labels[section] = label

            self.section_weights_layout.addRow(label, slider)

        section_weights_group.setLayout(self.section_weights_layout)
        form_layout.addRow(section_weights_group)

        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def _create_text_processing_settings_group(self):
        group_box = QGroupBox("Metin İşleme Ayarları")
        form_layout = QFormLayout()

        self.stopwords_checkbox = QCheckBox("Durdurma Kelimelerini Kaldır")
        self.lemmatization_checkbox = QCheckBox("Lematizasyon Yap")

        form_layout.addRow(self.stopwords_checkbox)
        form_layout.addRow(self.lemmatization_checkbox)

        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def _create_ui_settings_group(self):
        group_box = QGroupBox("Kullanıcı Arayüzü Ayarları")
        form_layout = QFormLayout()

        font_layout = QHBoxLayout()
        self.font_button = QPushButton("Font Seç ...")
        self.font_button.clicked.connect(self._choose_font)
        self.font_info_label = QLabel()
        self._update_font_label()

        font_layout.addWidget(self.font_button)
        font_layout.addWidget(self.font_info_label)
        font_layout.addStretch()

        form_layout.addRow("Uygulama Fontu:", font_layout)
        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def _create_buttons(self):
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Kaydet")
        self.save_button.setProperty("class", "success")
        self.save_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("İptal")
        self.cancel_button.setProperty("class", "secondary")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(button_layout)

    def _update_font_label(self):
        """Font Bilgi Etiketini Güncelle"""
        font_name = self.selected_font.family()
        font_size = self.selected_font.pointSize()
        self.font_info_label.setText(f"{font_name}, {font_size}pt")

    def _choose_font(self):
        """Font seçici dialogunu aç"""
        font, ok = QFontDialog.getFont(self.selected_font, self)
        if ok:
            self.selected_font = font
            self._update_font_label()

    def _load_settings_to_ui(self):
        self.min_similarity_spinbox.setValue(self.settings.get('minimum_similarity_threshold', 50))
        self.stopwords_checkbox.setChecked(self.settings.get('stopwords_enabled', True))
        self.lemmatization_checkbox.setChecked(self.settings.get('lemmatization_enabled', False))

        section_weights = self.settings.get('section_weights', {})
        for section, slider in self.section_sliders.items():
            weight = section_weights.get(section, 0.1)
            slider.setValue(int(weight * 100) if weight < 1 else int(weight))
        self._update_section_weights_display()

    def _update_section_weights_display(self):
        for section, slider in self.section_sliders.items():
            label = self.section_labels[section]
            label.setText(f"{section.replace('_', ' ').title()}: {slider.value()}%")

    def get_settings(self) -> Dict[str, Any]:
        self.settings['minimum_similarity_threshold'] = self.min_similarity_spinbox.value()
        self.settings['stopwords_enabled'] = self.stopwords_checkbox.isChecked()
        self.settings['lemmatization_enabled'] = self.lemmatization_checkbox.isChecked()
        self.settings['font'] = self.selected_font

        # Bölüm ağırlıklarını normalize et
        current_section_values = {s: slider.value() for s, slider in self.section_sliders.items()}
        total_weight = sum(current_section_values.values())

        if total_weight > 0:
            for section in current_section_values:
                self.settings['section_weights'][section] = current_section_values[section] / 100.0

        return self.settings


# -----------------------------
# Thread'ler
# -----------------------------
class NERAnalysisWorker(QThread):
    """NER analizi için ayrı thread"""
    progress_updated = pyqtSignal(int)
    analysis_completed = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self, reference_pdf_path, target_folder_path, settings: Dict[str, Any]):
        super().__init__()
        self.reference_pdf_path = reference_pdf_path
        self.target_folder_path = target_folder_path
        self.settings = settings

    def run(self):
        try:
            self.status_updated.emit("Analiz başlatılıyor...")
            analyzer = CVAnalyzer(settings=self.settings)
            self.progress_updated.emit(5)

            # Referans CV'yi yükle
            try:
                reference_cv_data = analyzer.extract_cv_data(self.reference_pdf_path)
                if not reference_cv_data:
                    self.error_occurred.emit("Referans CV verisi boş veya anlaşılamadı.")
                    return
            except Exception as e:
                self.error_occurred.emit(f"Referans CV yüklenemedi: {e}")
                return

            self.status_updated.emit("Referans CV işlendi. Hedef klasör taranıyor...")
            self.progress_updated.emit(10)

            # CV dosyalarını listele
            cv_files = [os.path.join(self.target_folder_path, f)
                        for f in os.listdir(self.target_folder_path)
                        if f.lower().endswith(('.pdf', '.docx', '.txt'))]

            if not cv_files:
                self.error_occurred.emit("Hedef klasörde analiz edilecek CV bulunamadı.")
                return

            total_cvs = len(cv_files)
            results = []

            # Her CV'yi analiz et
            for i, cv_file in enumerate(cv_files):
                self.progress_updated.emit(10 + int((i / total_cvs) * 80))
                self.status_updated.emit(f"{os.path.basename(cv_file)} analiz ediliyor...")

                try:
                    analysis_result = analyzer.analyze_cv(reference_cv_data, cv_file)
                    analysis_result['compared_cv_path'] = cv_file
                    results.append(analysis_result)
                except Exception as e:
                    print(f"CV analiz hatası ({os.path.basename(cv_file)}): {e}")
                    error_result = {
                        'file_name': os.path.basename(cv_file), 'person_name': 'Analiz Hatası',
                        'overall_similarity': 0.0, 'matches': {}, 'cv_data': None,
                        'compared_cv_path': cv_file, 'error': str(e),
                        'formatted_summary': f"Analiz hatası: {e}",
                        'is_suitable': False
                    }
                    results.append(error_result)

            self.progress_updated.emit(100)
            self.status_updated.emit("Analiz tamamlandı.")
            self.analysis_completed.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Genel analiz hatası: {e}")


class PromptAnalysisWorker(QThread):
    """Prompt'a göre analiz için ayrı thread"""
    progress_updated = pyqtSignal(int)
    analysis_completed = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self, prompt_text, target_folder_path, parent=None):
        super().__init__(parent)
        self.prompt_text = prompt_text
        self.target_folder_path = target_folder_path

    def run(self):
        try:
            self.status_updated.emit("Prompt'a göre analiz başlatılıyor...")
            self.progress_updated.emit(5)

            self.status_updated.emit("Prompt metni işleniyor...")
            prompt_embedding = model.encode(clean_text(self.prompt_text), convert_to_tensor=True)
            self.progress_updated.emit(10)

            results = analyze_cv_folder(prompt_embedding, 0.5, self.target_folder_path)

            if isinstance(results, tuple):
                results = [results]

            self.progress_updated.emit(100)
            self.status_updated.emit("Prompt'a göre analiz tamamlandı.")
            self.analysis_completed.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Prompt analiz hatası: {str(e)}")


class PromptAnalysisWindow(QWidget):
    """Prompt tabanlı analiz için özel arayüz penceresi."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setParent(parent)  # Parent'ı açıkça belirle
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.folder_path = None
        self._selected_cv_path = None
        self._setup_ui()

    def _setup_ui(self):
        # === Üst bar: sağ üstte geri dön butonu ===
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(6)
        top_bar.addStretch()
        self.back_button = QPushButton("← Ana Ekrana Dön")
        self.back_button.setFixedHeight(28)
        self.back_button.setStyleSheet("padding: 2px 8px;")
        self.back_button.setToolTip("Ana karşılaştırma ekranına geri dön")
        top_bar.addWidget(self.back_button, 0, Qt.AlignRight)
        self.layout.addLayout(top_bar)

        # === Prompt metni ===
        self.prompt_input_label = QLabel(
            "Analiz için bir Prompt girin (Örn: 'Python, C++ ve Bulut teknolojilerinde deneyimli yazılım geliştirici')"
        )
        self.layout.addWidget(self.prompt_input_label)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Prompt metnini buraya yazın...")
        self.prompt_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.prompt_input, 1)

        # === Klasör seç butonu (ORTADA & küçük) ===
        self.folder_select_button = QPushButton("Analiz Edilecek Klasörü Seç...")
        self.folder_select_button.setFixedSize(220, 30)
        self.folder_select_button.setStyleSheet("padding: 2px 8px;")

        folder_row = QHBoxLayout()
        folder_row.addStretch()
        folder_row.addWidget(self.folder_select_button)
        folder_row.addStretch()
        self.layout.addLayout(folder_row)

        # Seçilen klasör etiketi (ortada)
        self.folder_path_label = QLabel("Seçilen Klasör: Yok")
        self.layout.addWidget(self.folder_path_label, 0, Qt.AlignCenter)

        # === Analizi Başlat butonu (ORTADA & küçük) ===
        self.start_button = QPushButton("Analizi Başlat")
        self.start_button.setEnabled(False)
        self.start_button.setFixedSize(180, 30)
        self.start_button.setStyleSheet("padding: 4px 10px;")

        start_row = QHBoxLayout()
        start_row.addStretch()
        start_row.addWidget(self.start_button)
        start_row.addStretch()
        self.layout.addLayout(start_row)

        # === İlerleme & Durum ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Durum: Hazır")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        # === Sonuçlar ===
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(['Dosya Adı', 'Benzerlik Skoru', 'Durum'])
        self.results_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.layout.addWidget(self.results_tree, 2)

        # === Aksiyon butonları (başta pasif) ===
        self.btn_open_cv = QPushButton("Seçili CV’yi Aç")
        self.btn_view_pic = QPushButton("Resmi Görüntüle")
        self.btn_show_summary = QPushButton("Seçili CV Özeti")
        for b in (self.btn_open_cv, self.btn_view_pic, self.btn_show_summary):
            b.setEnabled(False)
            b.setFixedHeight(28)
            b.setStyleSheet("padding: 2px 8px;")

        btns = QHBoxLayout()
        btns.setSpacing(6)
        btns.setContentsMargins(0, 0, 0, 0)
        btns.addStretch()
        btns.addWidget(self.btn_open_cv)
        btns.addWidget(self.btn_view_pic)
        btns.addWidget(self.btn_show_summary)
        self.layout.addLayout(btns)

        # === Özet paneli (varsayılan kapalı) ===
        self.summary_group = QGroupBox("Seçili CV Özeti")
        self.summary_group.hide()
        _sv = QVBoxLayout(self.summary_group)
        self.summary_view = QTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setObjectName("cvSummary")
        self.summary_view.setPlaceholderText("Bir sonuç seçip 'Seçili CV Özeti' butonuna tıklayın.")
        _sv.addWidget(self.summary_view)
        self.layout.addWidget(self.summary_group)

        # === Sinyaller ===
        self.prompt_input.textChanged.connect(self._check_inputs)
        self.folder_select_button.clicked.connect(self._select_folder)
        self.results_tree.itemSelectionChanged.connect(self._on_prompt_result_selected)
        self.btn_open_cv.clicked.connect(self._open_selected_cv)
        self.btn_view_pic.clicked.connect(self._view_selected_cv_picture)
        self.btn_show_summary.clicked.connect(self._show_selected_cv_summary)
        self.back_button.clicked.connect(self._go_back_to_main)


    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Klasör Seç")
        if folder:
            self.folder_path = folder
            self.folder_path_label.setText(f"Seçilen Klasör: {folder}")
            self._check_inputs()

    def _check_inputs(self):
        self.start_button.setEnabled(bool(self.prompt_input.toPlainText().strip() and self.folder_path))

    def _on_prompt_result_selected(self):
        self._selected_cv_path = None
        for b in (self.btn_open_cv, self.btn_view_pic, self.btn_show_summary):
            b.setEnabled(False)
        self.summary_group.hide()

        item = self.results_tree.currentItem()
        if not item or not self.folder_path:
            return
        file_name = item.text(0)
        self._selected_cv_path = os.path.join(self.folder_path, file_name)
        for b in (self.btn_open_cv, self.btn_view_pic, self.btn_show_summary):
            b.setEnabled(True)

    def _open_selected_cv(self):
        if not self._selected_cv_path:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._selected_cv_path))

    def _view_selected_cv_picture(self):
        if not self._selected_cv_path:
            return
        img_bytes = extract_profile_picture_from_pdf_enhanced(self._selected_cv_path)
        if not img_bytes:
            QMessageBox.information(self, "Bilgi", "Profil resmi bulunamadı.")
            return
        pix = QPixmap()
        pix.loadFromData(img_bytes)
        lbl = QLabel()
        lbl.setPixmap(pix)
        lbl.setScaledContents(True)
        lbl.setMinimumSize(360, 360)
        dlg = QDialog(self)
        dlg.setWindowTitle("Profil Resmi")
        lay = QVBoxLayout(dlg)
        lay.addWidget(lbl)
        dlg.resize(480, 480)
        dlg.exec_()

    def _go_back_to_main(self):
        """Ana karşılaştırma ekranına geri dön"""
        # Parent'ı bulup ana ekrana dön
        parent_window = self.parent()
        while parent_window and not hasattr(parent_window, 'show_main_comparison_view'):
            parent_window = parent_window.parent()
        
        if parent_window and hasattr(parent_window, 'show_main_comparison_view'):
            parent_window.show_main_comparison_view()

    def _show_selected_cv_summary(self):
        if not self._selected_cv_path:
            return
        try:
            analyzer = CVAnalyzer()
            cv_data = analyzer.extract_cv_data(self._selected_cv_path)
            name = (cv_data.get('personal_info', {}) or {}).get('full_name') or cv_data.get('full_name') or "Bilinmiyor"
            email = (cv_data.get('personal_info', {}) or {}).get('email') or cv_data.get('email') or "Belirtilmemiş"
            phone = (cv_data.get('personal_info', {}) or {}).get('phone') or cv_data.get('phone') or "Belirtilmemiş"

            html = f"""
            <div style="line-height:1.5;">
              <h3 style="margin:0 0 8px 0;">{os.path.basename(self._selected_cv_path)}</h3>
              <p><b>Ad Soyad:</b> {name}</p>
              <p><b>E‑posta:</b> {email} &nbsp; | &nbsp; <b>Telefon:</b> {phone}</p>
            </div>
            """
            self.summary_view.setHtml(html)
            self.summary_group.show()
        except Exception as e:
            QMessageBox.warning(self, "Özet Hatası", str(e))


# -----------------------------
# Görüntüleme tercihleri
# -----------------------------
class ViewPreferencesDialog(QDialog):
    """Görüntüleme tercihleri dialogu."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Görüntüleme Tercihleri")
        self.setGeometry(300, 300, 400, 300)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self._create_widgets()

    def _create_widgets(self):
        self._create_result_count_group()
        self._create_filter_group()
        self._create_sort_group()
        self._create_buttons()

    def _create_result_count_group(self):
        group_box = QGroupBox("Görüntülenecek Sonuç Sayısı")
        vbox = QVBoxLayout()

        self.all_results_radio = QRadioButton("Tüm Sonuçlar")
        self.top5_radio = QRadioButton("En Benzer İlk 5 CV")
        self.top10_radio = QRadioButton("En Benzer İlk 10 CV")
        self.top20_radio = QRadioButton("En Benzer İlk 20 CV")
        self.custom_radio = QRadioButton("Özel Sayı:")

        self.custom_count_spinbox = QSpinBox()
        self.custom_count_spinbox.setRange(1, 999)
        self.custom_count_spinbox.setEnabled(False)
        self.custom_radio.toggled.connect(self.custom_count_spinbox.setEnabled)

        vbox.addWidget(self.all_results_radio)
        vbox.addWidget(self.top5_radio)
        vbox.addWidget(self.top10_radio)
        vbox.addWidget(self.top20_radio)

        custom_layout = QHBoxLayout()
        custom_layout.addWidget(self.custom_radio)
        custom_layout.addWidget(self.custom_count_spinbox)
        custom_layout.addStretch()
        vbox.addLayout(custom_layout)

        group_box.setLayout(vbox)
        self.layout.addWidget(group_box)
        self.all_results_radio.setChecked(True)

    def _create_filter_group(self):
        group_box = QGroupBox("Filtreleme Seçenekleri")
        vbox = QVBoxLayout()

        self.only_suitable_checkbox = QCheckBox("Sadece Uygun CV'leri Göster")
        self.min_score_checkbox = QCheckBox("Minimum Benzerlik Skoruna Göre Filtrele")

        self.min_score_spinbox = QSpinBox()
        self.min_score_spinbox.setRange(0, 100)
        self.min_score_spinbox.setValue(30)
        self.min_score_spinbox.setEnabled(False)
        self.min_score_checkbox.toggled.connect(self.min_score_spinbox.setEnabled)

        min_score_layout = QHBoxLayout()
        min_score_layout.addWidget(self.min_score_checkbox)
        min_score_layout.addWidget(QLabel("Min Skor (%):"))
        min_score_layout.addWidget(self.min_score_spinbox)
        min_score_layout.addStretch()

        vbox.addWidget(self.only_suitable_checkbox)
        vbox.addLayout(min_score_layout)

        group_box.setLayout(vbox)
        self.layout.addWidget(group_box)

    def _create_sort_group(self):
        group_box = QGroupBox("Sıralama Seçeneği")
        form_layout = QFormLayout()

        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Benzerlik Skoruna Göre (Azalan)")
        self.sort_combo.addItem("Benzerlik Skoruna Göre (Artan)")
        self.sort_combo.addItem("Dosya Adına Göre (A-Z)")
        self.sort_combo.addItem("Dosya Adına Göre (Z-A)")
        self.sort_combo.addItem("Duruma Göre (Uygun Önce)")

        form_layout.addRow("Sırala:", self.sort_combo)
        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def _create_buttons(self):
        button_box = QHBoxLayout()
        ok_button = QPushButton("Uygula")
        cancel_button = QPushButton("İptal")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_box.addStretch()
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        self.layout.addLayout(button_box)

    def get_preferences(self) -> Dict[str, Any]:
        preferences = {}
        if self.all_results_radio.isChecked():
            preferences['result_count'] = 0
        elif self.top5_radio.isChecked():
            preferences['result_count'] = 5
        elif self.top10_radio.isChecked():
            preferences['result_count'] = 10
        elif self.top20_radio.isChecked():
            preferences['result_count'] = 20
        elif self.custom_radio.isChecked():
            preferences['result_count'] = self.custom_count_spinbox.value()

        preferences['only_suitable'] = self.only_suitable_checkbox.isChecked()
        preferences['min_score_filter'] = self.min_score_checkbox.isChecked()
        preferences['min_score'] = self.min_score_spinbox.value()
        preferences['sort_option'] = self.sort_combo.currentIndex()

        return preferences


# -----------------------------
# Ana Pencere
# -----------------------------
class CVComparisonGUI(QMainWindow):
    """Ana GUI penceresi."""
    DEFAULT_SETTINGS = {
        'minimum_similarity_threshold': 50,
        'skills_weights': {'technical': 0.7, 'soft': 0.3},
        'section_weights': {
            'personal_info': 0.1, 'education': 0.2, 'experience': 0.4,
            'skills': 0.2, 'projects': 0.05, 'certifications': 0.05
        },
        'stopwords_enabled': True,
        'lemmatization_enabled': False,
        'dark_theme': False,
        'font': QFont('Segoe UI', 10),
        'language_priority': 0,
        'minimum_experience_years': 0,
        'require_degree': False,
        'require_experience': False,
        'required_Skills': '',
        'education_level_filter': 0,
        'recent_experience_only': False,
        'case_sensitive': False,
        'max_threads': 4,
        'cache_results': True,
        'show_selected_cv_summary': False  # ÖZET VARSAYILAN KAPALI
    }

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.setWindowTitle("CV Analiz Motoru")
        self.setGeometry(100, 100, 1400, 900)

        self.reference_pdf_path = ""
        self.target_folder_path = ""
        self.analysis_results = []
        self.current_selected_cv_path = None
        self.current_selected_result = None
        self.current_view_preferences = {
            'result_count': 0, 'only_suitable': False,
            'min_score_filter': False, 'min_score': 30, 'sort_option': 0
        }
        self.settings = settings or self.DEFAULT_SETTINGS.copy()

        # QStackedWidget kullanarak görünümler arası geçiş
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.apply_theme_and_font()
        self.create_menu_bar()
        
        # Ana karşılaştırma UI'sını oluştur
        self.main_comparison_widget = QWidget()
        self.setup_main_comparison_ui()
        
        # Prompt analiz widget'ını oluştur - parent olarak self'i ver
        self.prompt_analysis_widget = PromptAnalysisWindow(self)
        self.prompt_analysis_widget.start_button.clicked.connect(self.start_prompt_analysis)
        
        # Stack'e ekle
        self.stacked_widget.addWidget(self.main_comparison_widget)
        self.stacked_widget.addWidget(self.prompt_analysis_widget)
        
        # Başlangıçta ana görünümü göster
        self.stacked_widget.setCurrentWidget(self.main_comparison_widget)

    # ----------------- Tema & Font -----------------
    def apply_theme_and_font(self):
        """Tema ve fontu ayarlarıyla birlikte uygular."""
        font = self.settings.get('font', self.DEFAULT_SETTINGS['font'])
        QApplication.instance().setFont(font)

        font_family = font.family()
        font_size = font.pointSize()

        if self.settings['dark_theme']:
            stylesheet = theme_system.ThemeManager.get_dark_theme_stylesheet(font_family, font_size)
        else:
            stylesheet = theme_system.ThemeManager.get_light_theme_stylesheet(font_family, font_size)

        # Hover efektleri
        stylesheet += """
            QPushButton:hover, QToolButton:hover { 
                background-color: #e0e0e0; border: 1px solid #7c7c7c; 
            }
            QPushButton:pressed, QToolButton:pressed { 
                background-color: #cccccc; 
            }
            QMenu::item:hover { 
                background-color: #e0e0e0; color: black; 
            }
        """
        self.setStyleSheet(stylesheet)
        self.update_widgets_font(font)

    def update_widgets_font(self, font):
        """Tüm widget'ların fontunu özyineli olarak günceller."""
        def update_font_recursive(widget):
            if widget:
                widget.setFont(font)
                for child in widget.findChildren(QWidget):
                    if child.parent() == widget:
                        child.setFont(font)
        update_font_recursive(self)

    # ----------------- Ana UI -----------------
    def setup_main_comparison_ui(self):
        """Ana karşılaştırma arayüzünü oluşturan metot."""
        main_layout = QVBoxLayout()
        self.main_comparison_widget.setLayout(main_layout)

        # Üst satır: Dosya/klasör seçimi + sağda "Prompt'a göre Analiz"
        selection_frame = QFrame()
        selection_layout = QHBoxLayout()
        selection_frame.setLayout(selection_layout)

        self.source_path_edit = QLineEdit("Referans CV seçilmedi.")
        self.source_path_edit.setReadOnly(True)
        self.select_source_button = QPushButton("Referans CV Seç...")
        self.select_source_button.clicked.connect(self.select_reference_cv)

        self.target_path_edit = QLineEdit("Hedef klasör seçilmedi.")
        self.target_path_edit.setReadOnly(True)
        self.select_target_button = QPushButton("Hedef Klasör Seç...")
        self.select_target_button.clicked.connect(self.select_target_folder)

        selection_layout.addWidget(self.select_source_button)
        selection_layout.addWidget(self.source_path_edit)
        selection_layout.addWidget(self.select_target_button)
        selection_layout.addWidget(self.target_path_edit)

        # Sağ üst köşeye buton
        selection_layout.addStretch()
        self.prompt_button = QPushButton("Prompt'a göre Analiz")
        self.prompt_button.setToolTip("Prompt tabanlı analiz ekranını aç")
        self.prompt_button.clicked.connect(self.show_prompt_analysis_view)
        selection_layout.addWidget(self.prompt_button, 0, Qt.AlignRight)

        main_layout.addWidget(selection_frame)

        # Analiz ve Durum
        status_frame = QFrame()
        status_layout = QVBoxLayout()
        status_frame.setLayout(status_layout)

        self.start_button = QPushButton("Analizi Başlat")
        self.start_button.setEnabled(False)
        self.start_button.setFixedSize(200, 40)
        self.start_button.clicked.connect(self.start_analysis)
        self.start_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        status_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        status_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Durum: Hazır")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        main_layout.addWidget(status_frame)

        # Sonuçlar & Özet splitter
        results_splitter = QSplitter(Qt.Horizontal)

        # Sol: Sonuçlar
        results_group = QGroupBox("Karşılaştırma Sonuçları")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(['Dosya Adı', 'Benzerlik Skoru', 'Durum'])
        self.results_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_tree.setSortingEnabled(True)
        self.results_tree.itemClicked.connect(self._on_cv_selected)
        results_layout.addWidget(self.results_tree)

        button_layout = QHBoxLayout()
        self.btn_open_cv_file = QPushButton("Seçili CV Dosyasını Aç")
        self.btn_open_cv_file.setEnabled(False)
        self.btn_open_cv_file.clicked.connect(self.open_selected_cv_file)

        self.show_profile_picture_button = QPushButton("Profil Resmini Göster")
        self.show_profile_picture_button.setEnabled(False)
        self.show_profile_picture_button.clicked.connect(self.show_selected_cv_picture)

        button_layout.addWidget(self.btn_open_cv_file)
        button_layout.addWidget(self.show_profile_picture_button)
        button_layout.addStretch()
        results_layout.addLayout(button_layout)

        # Sağ: Özet Paneli (varsayılan kapalı, koyu temaya uyum için objectName)
        self.summary_group = QGroupBox("Seçili CV Özeti")
        summary_layout = QVBoxLayout()
        self.summary_group.setLayout(summary_layout)

        self.cv_summary_display = QTextEdit()
        self.cv_summary_display.setReadOnly(True)
        self.cv_summary_display.setPlaceholderText("Bir CV seçtiğinizde, özet bilgileri burada görünecek...")
        self.cv_summary_display.setObjectName("cvSummary")  # tema ile uyum

        summary_layout.addWidget(self.cv_summary_display)

        # Splitter'a ekle
        results_splitter.addWidget(results_group)
        results_splitter.addWidget(self.summary_group)
        results_splitter.setSizes([700, 500])
        main_layout.addWidget(results_splitter, 1)

        # Özet varsayılan görünümü (ayar)
        if not self.settings.get('show_selected_cv_summary', False):
            self.summary_group.hide()

        self._check_if_ready()

    # ----------------- Seçim/Özet -----------------
    def _on_cv_selected(self, item: QTreeWidgetItem, column: int):
        """CV seçildiğinde çağrılan fonksiyon"""
        if item.parent() is None:
            file_name = item.text(0)
            found_result = next((r for r in self.analysis_results if r.get('file_name') == file_name), None)
            if found_result:
                self.current_selected_cv_path = found_result.get('compared_cv_path')
                self.current_selected_result = found_result
                self.btn_open_cv_file.setEnabled(True)
                self.show_profile_picture_button.setEnabled(True)

                # CV özetini göster
                self._display_cv_summary(found_result)
                self.statusBar().showMessage(f"{file_name} seçildi.")
                # Özet kapalıysa da kullanıcı seçtiyse gösterebilir: otomatik açmak istemiyorsanız yoruma alın
                if self.settings.get('show_selected_cv_summary', False):
                    self.summary_group.show()
            else:
                self._clear_cv_selection()
                self.statusBar().showMessage(f"'{file_name}' için dosya yolu bulunamadı.")
        else:
            self._on_cv_selected(item.parent(), column)

    def _clear_cv_selection(self):
        """CV seçimini temizle"""
        self.current_selected_cv_path = None
        self.current_selected_result = None
        self.btn_open_cv_file.setEnabled(False)
        self.show_profile_picture_button.setEnabled(False)
        self.cv_summary_display.clear()

    def _get_safe_value(self, data, key, default="Belirtilmemiş"):
        if not isinstance(data, dict):
            return default
        value = data.get(key, default)
        return value if value and str(value).strip() else default

    def _get_list_as_string(self, data_list):
        if not data_list:
            return "Belirtilmemiş"
        if isinstance(data_list, list):
            valid_items = [str(item).strip() for item in data_list if item and str(item).strip()]
            return ", ".join(valid_items) if valid_items else "Belirtilmemiş"
        return str(data_list).strip() if data_list else "Belirtilmemiş"

    def _extract_section_data_by_headers(self, cv_data: Dict[str, Any], section_type: str) -> List[str]:
        """CV başlıklarına göre dinamik veri çıkarma (geliştirilmiş)"""
        section_data = []
        cv_text = ""
        if isinstance(cv_data, dict) and 'raw_text' in cv_data:
            cv_text = cv_data['raw_text']
        elif hasattr(cv_data, 'get') and cv_data.get('text'):
            cv_text = cv_data['text']
        if not cv_text:
            return section_data

        header_patterns = {
            'experience': [
                r'(?i)(İŞ DENEYİMİ|WORK EXPERIENCE|EXPERIENCE|DENEYİM|PROFESSIONAL EXPERIENCE|CAREER|İŞ GEÇMİŞİ)',
                r'(?i)(EMPLOYMENT|EMPLOYMENT HISTORY|WORK HISTORY|PROFESSIONAL BACKGROUND|CAREER HISTORY)'
            ],
            'education': [
                r'(?i)(EĞİTİM|EDUCATION|EDUCATIONAL BACKGROUND|ACADEMIC|ÖĞRENİM|OKUL)',
                r'(?i)(ACADEMIC QUALIFICATIONS|QUALIFICATIONS|UNIVERSITY|ÜNİVERSİTE|EĞİTİM BİLGİLERİ)'
            ],
            'skills': [
                r'(?i)(BECERİLER|SKILLS|TECHNICAL SKILLS|YETENEKLER|EXPERTISE|COMPETENCIES)',
                r'(?i)(ABILITIES|CAPABILITIES|PROFICIENCIES|TOOLS|TECHNOLOGIES)'
            ],
            'projects': [
                r'(?i)(PROJELER|PROJECTS|PROJECT EXPERIENCE|PORTFOLIO|WORKS)',
            ],
            'certifications': [
                r'(?i)(SERTİFİKALAR|CERTIFICATES|CERTIFICATIONS|LICENSES|LİSANSLAR)',
            ]
        }

        patterns = header_patterns.get(section_type, [])
        for pattern in patterns:
            matches = re.search(pattern, cv_text)
            if matches:
                start_pos = matches.end()
                next_header_pattern = r'(?i)\n\s*([A-ZÇĞÖŞÜİ\s]{3,}|[A-Z\s]{3,})\s*\n'
                next_match = re.search(next_header_pattern, cv_text[start_pos:])
                if next_match:
                    end_pos = start_pos + next_match.start()
                    section_text = cv_text[start_pos:end_pos]
                else:
                    section_text = cv_text[start_pos:start_pos + 2000]

                lines = [line.strip() for line in section_text.split('\n') if line.strip()]
                valid_lines = []
                for line in lines:
                    if len(line) < 5:
                        continue
                    if re.match(r'^[^\w]*', line):
                        continue
                    if re.match(r'^[\d\-/\s]+', line):
                        continue

                        continue
                    valid_lines.append(line)

                section_limit = 10 if section_type == 'experience' else 8
                section_data.extend(valid_lines[:section_limit])
                break

        return section_data

    def _get_translated_section_name(self, section_key: str) -> str:
        translations = {
            'personal_info': 'Kişisel Bilgiler',
            'education': 'Eğitim',
            'experience': 'İş Deneyimi',
            'skills': 'Beceriler',
            'projects': 'Projeler',
            'certifications': 'Sertifikalar',
            'languages': 'Diller',
            'references': 'Referanslar',
            'summary': 'Özet',
            'objective': 'Hedef',
            'interests': 'İlgi Alanları',
            'awards': 'Ödüller'
        }
        clean_key = section_key.replace('_match', '').replace('_', ' ')
        return translations.get(section_key.replace('_match', ''), clean_key.title())

    def _format_experience_data(self, experience_data: List[str]) -> str:
        if not experience_data:
            return "Belirtilmemiş"
        formatted_items = []
        current_item = ""
        for item in experience_data:
            if (any(keyword in item.lower() for keyword in ['ltd', 'inc', 'corp', 'company', 'şirket', 'a.ş.', 'ltd.şti.']) or
                any(title in item.lower() for title in ['developer', 'engineer', 'manager', 'analyst', 'specialist', 'geliştirici', 'mühendis', 'uzman'])):
                if current_item:
                    formatted_items.append(current_item.strip())
                current_item = item
            else:
                if current_item:
                    current_item += " " + item
                else:
                    current_item = item
        if current_item:
            formatted_items.append(current_item.strip())
        return "\n".join([f"• {item}" for item in formatted_items[:5]])

    def _format_education_data(self, education_data: List[str]) -> str:
        if not education_data:
            return "Belirtilmemiş"
        formatted_items = []
        for item in education_data:
            if len(item) > 15:
                formatted_items.append(f"• {item}")
        return "\n".join(formatted_items[:4])

    def _display_cv_summary(self, result: Dict[str, Any]):
        """Seçili CV'nin özetini oluştur ve göster"""
        try:
            if not result:
                self.cv_summary_display.setHtml("<p style='color: #999; text-align: center;'>Bir CV seçtiğinizde, özet bilgileri burada görünecek...</p>")
                return

            cv_data = result.get('cv_data', {})
            if not cv_data or not isinstance(cv_data, dict):
                error_html = """
                <div style="color: #dc3545; padding: 15px; text-align: center;">
                    <h3>⚠️ Veri Bulunamadı</h3>
                    <p>Bu CV için detaylı bilgi çıkarılamadı.</p>
                </div>
                """
                self.cv_summary_display.setHtml(error_html)
                return

            html_content = self._generate_cv_summary_html(result, cv_data)
            self.cv_summary_display.setHtml(html_content)

        except Exception as e:
            error_html = f"""
            <div style="color: #dc3545; padding: 15px; border: 1px solid #dc3545; border-radius: 5px; background-color: #f8d7da;">
                <h3>🚫 Özet Oluşturma Hatası</h3>
                <p>CV özeti oluşturulurken hata oluştu:</p>
                <p><code>{str(e)}</code></p>
                <p><small>Lütfen farklı bir CV seçmeyi deneyin.</small></p>
            </div>
            """
            self.cv_summary_display.setHtml(error_html)

    def _generate_cv_summary_html(self, result: Dict[str, Any], cv_data: Dict[str, Any]) -> str:
        """CV özeti için HTML içeriği (güncellenmiş)"""
        file_name = self._get_safe_value(result, 'file_name', 'Bilinmiyor')
        person_name = self._get_safe_value(result, 'person_name', 'Bilinmiyor')
        overall_similarity = result.get('overall_similarity', 0.0)
        is_suitable = result.get('is_suitable', False)

        email = self._get_safe_value(cv_data.get('personal_info', {}), 'email')
        phone = self._get_safe_value(cv_data.get('personal_info', {}), 'phone')

        if email == "Belirtilmemiş":
            email = self._get_safe_value(cv_data, 'email')
        if phone == "Belirtilmemiş":
            phone = self._get_safe_value(cv_data, 'phone')

        status_color = "#28a745" if is_suitable else "#dc3545"
        status_text = "✅ Uygun" if is_suitable else "❌ Uygun Değil"

        html_content = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; max-width: 100%;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h2 style="margin: 0; font-size: 18px;">📄 {file_name}</h2>
                <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">CV Özeti</p>
            </div>

            <div style="background: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #007bff;">
                <h3 style="color: #495057; margin: 0 0 8px 0; font-size: 14px;">👤 Kişisel Bilgiler</h3>
                <div style="display: grid; gap: 5px;">
                    <p style="margin: 3px 0;"><strong>Ad Soyad:</strong> {person_name}</p>
                    <p style="margin: 3px 0;"><strong>E-posta:</strong> {email}</p>
                    <p style="margin: 3px 0;"><strong>Telefon:</strong> {phone}</p>
                </div>
            </div>

            <div style="background: #e9ecef; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid {status_color};">
                <h3 style="color: #495057; margin: 0 0 8px 0; font-size: 14px;">📊 Analiz Sonucu</h3>
                <div style="display: grid; gap: 5px;">
                    <p style="margin: 3px 0;"><strong>Benzerlik Skoru:</strong> 
                    <span style="font-size: 16px; font-weight: bold; color: {status_color};">{overall_similarity:.2f}%</span>
                    </p>
                    <p style="margin: 3px 0;"><strong>Durum:</strong> 
                    <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
                    </p>
                </div>
            </div>
        """

        # Eğitim
        education_data = self._extract_section_data_by_headers(cv_data, 'education')
        if not education_data:
            education_info = cv_data.get('education', [])
            if education_info and isinstance(education_info, list):
                education_data = []
                for edu in education_info[:3]:
                    if isinstance(edu, dict):
                        degree = self._get_list_as_string(edu.get('degree', []))
                        university = self._get_list_as_string(edu.get('university', []))
                        department = self._get_list_as_string(edu.get('department', []))
                        if degree != "Belirtilmemiş" or university != "Belirtilmemiş":
                            parts = []
                            if degree != "Belirtilmemiş": parts.append(degree)
                            if department != "Belirtilmemiş": parts.append(department)
                            if university != "Belirtilmemiş": parts.append(f"({university})")
                            education_data.append(" ".join(parts))
        if education_data:
            formatted_education = self._format_education_data(education_data)
            html_content += f"""
            <div style="background: #d1ecf1; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #17a2b8;">
                <h3 style="color: #0c5460; margin: 0 0 8px 0; font-size: 14px;">🎓 Eğitim</h3>
                <div style="background: white; padding: 10px; border-radius: 4px; border-left: 3px solid #17a2b8;">
                    <div style='margin: 0; font-size: 13px; line-height: 1.6; white-space: pre-line;'>{formatted_education}</div>
                </div>
            </div>
            """

        # Deneyim
        experience_data = self._extract_section_data_by_headers(cv_data, 'experience')
        if not experience_data:
            experience_info = cv_data.get('experience', [])
            if experience_info and isinstance(experience_info, list):
                experience_data = []
                for exp in experience_info[:3]:
                    if isinstance(exp, dict):
                        company = self._get_list_as_string(exp.get('company', []))
                        title = self._get_list_as_string(exp.get('title', []))
                        duration = exp.get('duration_formatted', '')
                        if company != "Belirtilmemiş" or title != "Belirtilmemiş":
                            parts = []
                            if title != "Belirtilmemiş": parts.append(title)
                            if company != "Belirtilmemiş": parts.append(f"@ {company}")
                            if duration: parts.append(f"({duration})")
                            experience_data.append(" ".join(parts))
        if experience_data:
            formatted_experience = self._format_experience_data(experience_data)
            html_content += f"""
            <div style="background: #d4edda; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #28a745;">
                <h3 style="color: #155724; margin: 0 0 8px 0; font-size: 14px;">💼 İş Deneyimi</h3>
                <div style="background: white; padding: 10px; border-radius: 4px; border-left: 3px solid #28a745;">
                    <div style='margin: 0; font-size: 13px; line-height: 1.6; white-space: pre-line;'>{formatted_experience}</div>
                </div>
            </div>
            """

        # Beceriler
        skills_data = self._extract_section_data_by_headers(cv_data, 'skills')
        if not skills_data:
            skills = cv_data.get('skills', [])
            if skills and isinstance(skills, list):
                filtered = [s.strip() for s in skills if s and str(s).strip()]
                skills_data = filtered[:15]
        if skills_data:
            skills_text = ""
            for i, skill in enumerate(skills_data[:20]):
                if i > 0 and i % 5 == 0:
                    skills_text += "\n"
                skills_text += f"• {skill.strip()}"
                if i < len(skills_data) - 1 and (i + 1) % 5 != 0:
                    skills_text += "  "
            html_content += f"""
            <div style="background: #fff3cd; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #ffc107;">
                <h3 style="color: #856404; margin: 0 0 8px 0; font-size: 14px;">🛠️ Beceriler</h3>
                <div style="background: white; padding: 10px; border-radius: 4px;">
                    <div style="margin: 0; font-size: 13px; line-height: 1.6; white-space: pre-line;">{skills_text}</div>
                </div>
                <p style="margin: 5px 0 0 0; font-size: 11px; color: #856404;">Toplam {len(skills_data)} beceri tespit edildi</p>
            </div>
            """

        # Kapanış
        html_content += f"""
            <div style="text-align: center; margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 6px; border: 1px dashed #dee2e6;">
                <p style="margin: 0; font-size: 11px; color: #6c757d;">
                    📅 Özet oluşturulma zamanı: {QDateTime.currentDateTime().toString("dd.MM.yyyy hh:mm")}
                </p>
            </div>
        </div>
        """
        return html_content

    # ----------------- Menü -----------------
    def create_menu_bar(self):
        menubar = self.menuBar()

        # Dosya
        file_menu = menubar.addMenu("Dosya")
        file_menu.addAction(self._create_action("Referans CV Seç", self.select_reference_cv, "Ctrl+0"))
        file_menu.addAction(self._create_action("Hedef Klasör Seç", self.select_target_folder, "Ctrl+Shift+0"))
        file_menu.addSeparator()
        file_menu.addAction(self._create_action("Sonuçları Dışa Aktar...", self.export_results, "Ctrl+E"))
        file_menu.addSeparator()
        file_menu.addAction(self._create_action("Çıkış", self.close, "Ctrl+Q"))

        # Ayarlar
        settings_menu = menubar.addMenu("Ayarlar")
        settings_menu.addAction(self._create_action("Gelişmiş Ayarlar...", self.open_preferences, "Ctrl+,"))
        settings_menu.addSeparator()
        theme_action = self._create_action("Koyu Tema", self.toggle_theme)
        theme_action.setCheckable(True)
        theme_action.setChecked(self.settings['dark_theme'])
        settings_menu.addAction(theme_action)

        # Görünüm
        view_menu = menubar.addMenu("Görünüm")
        view_menu.addAction(self._create_action("Görüntüleme Tercihleri...", self.open_view_preferences, "Ctrl+Shift+P"))
        view_menu.addSeparator()
        view_menu.addAction(self._create_action("Ana Karşılaştırma", self.show_main_comparison_view))
        view_menu.addAction(self._create_action("Prompt'a göre Analiz", self.show_prompt_analysis_view))
        view_menu.addSeparator()
        self.toggle_summary_action = self._create_action("Özet Panelini Gizle/Göster", self.toggle_summary_panel, "Ctrl+D")
        view_menu.addAction(self.toggle_summary_action)

        # Yardım
        help_menu = menubar.addMenu("Yardım")
        help_menu.addAction(self._create_action("Klavye Kısayolları", self.show_shortcuts))
        help_menu.addSeparator()
        help_menu.addAction(self._create_action("Hakkında", self.show_about))

    def toggle_summary_panel(self):
        """Özet panelini göster/gizle (kalıcı ayar)"""
        try:
            if hasattr(self, 'summary_group'):
                will_show = not self.summary_group.isVisible()
                self.summary_group.setVisible(will_show)
                self.settings['show_selected_cv_summary'] = will_show  # kalıcı tercih
                self.statusBar().showMessage("Özet paneli gösterildi." if will_show else "Özet paneli gizlendi.")
        except Exception as e:
            self.statusBar().showMessage(f"Panel geçiş hatası: {str(e)}")

    def _create_action(self, text, slot, shortcut=None):
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def toggle_theme(self):
        self.settings['dark_theme'] = not self.settings['dark_theme']
        self.apply_theme_and_font()
        for action in self.menuBar().findChildren(QAction):
            if action.text() == "Koyu Tema":
                action.setChecked(self.settings['dark_theme'])
                break

    # ----------------- Görünüm tercihleri -----------------
    def open_view_preferences(self):
        dialog = ViewPreferencesDialog(self)
        self._set_dialog_preferences(dialog, self.current_view_preferences)
        if dialog.exec_() == QDialog.Accepted:
            self.current_view_preferences = dialog.get_preferences()
            self.apply_view_preferences()
            self.statusBar().showMessage("Görüntüleme tercihleri güncellendi.")

    def _set_dialog_preferences(self, dialog, preferences):
        if preferences['result_count'] == 0:
            dialog.all_results_radio.setChecked(True)
        elif preferences['result_count'] == 5:
            dialog.top5_radio.setChecked(True)
        elif preferences['result_count'] == 10:
            dialog.top10_radio.setChecked(True)
        elif preferences['result_count'] == 20:
            dialog.top20_radio.setChecked(True)
        else:
            dialog.custom_radio.setChecked(True)
            dialog.custom_count_spinbox.setValue(preferences['result_count'])

        dialog.only_suitable_checkbox.setChecked(preferences['only_suitable'])
        dialog.min_score_checkbox.setChecked(preferences['min_score_filter'])
        dialog.min_score_spinbox.setValue(preferences['min_score'])
        dialog.sort_combo.setCurrentIndex(preferences['sort_option'])

    def apply_view_preferences(self):
        if not self.analysis_results:
            self.results_tree.clear()
            return

        filtered_results = self.analysis_results.copy()

        # Filtreleme
        if self.current_view_preferences['only_suitable']:
            filtered_results = [r for r in filtered_results if r.get('is_suitable', False)]
        if self.current_view_preferences['min_score_filter']:
            min_score = self.current_view_preferences['min_score']
            filtered_results = [r for r in filtered_results if r.get('overall_similarity', 0) >= min_score]

        # Sıralama
        sort_option = self.current_view_preferences['sort_option']
        if sort_option == 0:
            filtered_results.sort(key=lambda x: x.get('overall_similarity', 0), reverse=True)
        elif sort_option == 1:
            filtered_results.sort(key=lambda x: x.get('overall_similarity', 0))
        elif sort_option == 2:
            filtered_results.sort(key=lambda x: x.get('file_name', ''))
        elif sort_option == 3:
            filtered_results.sort(key=lambda x: x.get('file_name', ''), reverse=True)
        elif sort_option == 4:
            filtered_results.sort(key=lambda x: (not x.get('is_suitable', False), -x.get('overall_similarity', 0)))

        # Limit
        if self.current_view_preferences['result_count'] > 0:
            filtered_results = filtered_results[:self.current_view_preferences['result_count']]

        self.display_results_in_tree(filtered_results)

    def display_results_in_tree(self, results: List[Dict[str, Any]]):
        self.results_tree.clear()
        self.results_tree.setSortingEnabled(False)

        for result in results:
            file_name = result.get('file_name', 'Bilinmiyor')
            overall_similarity = result.get('overall_similarity', 0.0)
            is_suitable = result.get('is_suitable', False)

            status_text = "Analiz Hatası" if 'error' in result else ("Uygun" if is_suitable else "Uygun Değil")
            status_color = QColor("darkred") if 'error' in result else (QColor("green") if is_suitable else QColor("red"))

            item = NumericTreeWidgetItem()
            item.setText(0, file_name)
            item.setText(1, f"{overall_similarity:.2f}%")
            item.setText(2, status_text)
            item.setData(1, Qt.UserRole, overall_similarity)
            item.setForeground(1, status_color)
            item.setForeground(2, status_color)

            matches = result.get('matches', {})
            for section_key, section_match_info in matches.items():
                if isinstance(section_match_info, dict):
                    section_score = section_match_info.get('score', 0.0)
                    section_matches_list = section_match_info.get('matches') or section_match_info.get('intersection', [])
                    section_item = NumericTreeWidgetItem(item)
                    section_item.setText(0, f"  {section_key.replace('_', ' ').title()}")
                    section_item.setText(1, f"{section_score:.2f}%")
                    section_item.setText(2, f"Eşleşenler: {', '.join(section_matches_list[:3])}" if section_matches_list else "Yok")
                    section_item.setData(1, Qt.UserRole, section_score)

            self.results_tree.addTopLevelItem(item)

    # ----------------- Dışa aktarma -----------------
    def export_results(self):
        if not self.analysis_results:
            QMessageBox.warning(self, "Veri Yok", "Dışa aktarılacak analiz sonucu bulunmuyor.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Sonuçları Kaydet", "", "CSV Dosyası (*.csv);;JSON Dosyası (*.json);;Metin Dosyası (*.txt)"
        )

        if not file_path:
            return

        try:
            if file_path.endswith('.csv'):
                self._export_to_csv(file_path)
            elif file_path.endswith('.json'):
                self._export_to_json(file_path)
            else:
                self._export_to_txt(file_path)
            QMessageBox.information(self, "Başarılı", f"Sonuçlar başarıyla kaydedildi:\n{file_path}")
            self.statusBar().showMessage(f"Sonuçlar dışa aktarıldı: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Kaydetme Hatası", f"Dosya kaydedilirken hata oluştu:\n{str(e)}")

    def _export_to_csv(self, file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dosya Adı', 'Kişi Adı', 'Genel Benzerlik (%)', 'Durum', 'E-posta', 'Telefon'])
            for result in self.analysis_results:
                cv_data = result.get('cv_data', {})
                personal_info = cv_data.get('personal_info', {}) if isinstance(cv_data, dict) else {}
                writer.writerow([
                    result.get('file_name', ''),
                    result.get('person_name', ''),
                    f"{result.get('overall_similarity', 0):.2f}",
                    "Uygun" if result.get('is_suitable', False) else "Uygun Değil",
                    personal_info.get('email', '') or cv_data.get('email', '') if isinstance(cv_data, dict) else '',
                    personal_info.get('phone', '') or cv_data.get('phone', '') if isinstance(cv_data, dict) else ''
                ])

    def _export_to_json(self, file_path):
        export_data = []
        for result in self.analysis_results:
            export_data.append({
                'file_name': result.get('file_name', ''),
                'person_name': result.get('person_name', ''),
                'overall_similarity': result.get('overall_similarity', 0),
                'is_suitable': result.get('is_suitable', False),
                'matches': result.get('matches', {}),
                'cv_data': result.get('cv_data', {})
            })
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, ensure_ascii=False, indent=2)

    def _export_to_txt(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as txtfile:
            txtfile.write("CV KARŞILAŞTIRMA SONUÇLARI\n" + "=" * 50 + "\n\n")
            for result in self.analysis_results:
                txtfile.write(f"Dosya: {result.get('file_name', '')}\n")
                txtfile.write(f"Kişi: {result.get('person_name', '')}\n")
                txtfile.write(f"Benzerlik: {result.get('overall_similarity', 0):.2f}%\n")
                txtfile.write(f"Durum: {'Uygun' if result.get('is_suitable', False) else 'Uygun Değil'}\n")
                txtfile.write("-" * 50 + "\n\n")

    # ----------------- Ekranlar arası geçiş -----------------
    def show_prompt_analysis_view(self):
        """Prompt analiz görünümünü göster"""
        self.stacked_widget.setCurrentWidget(self.prompt_analysis_widget)
        self.prompt_analysis_widget.results_tree.clear()
        self.prompt_analysis_widget.status_label.setText("Durum: Hazır")
        self.prompt_analysis_widget.progress_bar.setValue(0)

    def show_main_comparison_view(self):
        """Ana karşılaştırma görünümünü göster"""
        self.stacked_widget.setCurrentWidget(self.main_comparison_widget)

    # ----------------- Prompt analizi -----------------
    def start_prompt_analysis(self):
        prompt_text = self.prompt_analysis_widget.prompt_input.toPlainText()
        folder_path = self.prompt_analysis_widget.folder_path

        if not prompt_text or not folder_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir prompt metni girin ve klasör seçin.")
            return

        self.prompt_analysis_widget.start_button.setEnabled(False)
        self.prompt_analysis_widget.status_label.setText("Durum: Analiz başlatılıyor...")
        self.prompt_analysis_widget.results_tree.clear()

        self.worker_thread = PromptAnalysisWorker(prompt_text, folder_path)
        self.worker_thread.progress_updated.connect(self.prompt_analysis_widget.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.prompt_analysis_widget.status_label.setText)
        self.worker_thread.error_occurred.connect(self.handle_error_in_prompt_view)
        self.worker_thread.analysis_completed.connect(self.display_prompt_results)
        self.worker_thread.start()

    def display_prompt_results(self, results: list):
        self.prompt_analysis_widget.status_label.setText("Analiz tamamlandı.")
        self.prompt_analysis_widget.start_button.setEnabled(True)
        self.prompt_analysis_widget.results_tree.clear()

        if isinstance(results, list):
            results.sort(key=lambda x: x[1] if len(x) > 1 else 0, reverse=True)

            for result in results:
                if isinstance(result, (list, tuple)) and len(result) >= 3:
                    file_name, score, status = result[0], result[1], result[2]
                    item = QTreeWidgetItem([str(file_name), f"{float(score):.2f}", str(status)])
                    self.prompt_analysis_widget.results_tree.addTopLevelItem(item)
                else:
                    item = QTreeWidgetItem([f"Hata: {str(result)}", "", ""])
                    item.setForeground(0, QColor("darkred"))
                    self.prompt_analysis_widget.results_tree.addTopLevelItem(item)

    def handle_error_in_prompt_view(self, message: str):
        QMessageBox.critical(self, "Hata", message)
        self.prompt_analysis_widget.start_button.setEnabled(True)
        self.prompt_analysis_widget.progress_bar.setValue(0)
        self.prompt_analysis_widget.status_label.setText(f"Durum: Hata - {message}")

    # ----------------- Dosya işlemleri -----------------
    def select_reference_cv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Referans CV Seç", "",
            "PDF Dosyaları (*.pdf);;Word Dosyaları (*.docx);;Metin Dosyaları (*.txt)"
        )
        if file_path:
            self.reference_pdf_path = file_path
            self.source_path_edit.setText(os.path.basename(file_path))
            self.statusBar().showMessage(f"Referans CV seçildi: {os.path.basename(file_path)}")
            self._check_if_ready()

    def select_target_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Hedef Klasörü Seç")
        if folder_path:
            self.target_folder_path = folder_path
            self.target_path_edit.setText(os.path.basename(folder_path))
            self.statusBar().showMessage(f"Hedef klasör seçildi: {os.path.basename(folder_path)}")
            self._check_if_ready()

    def _check_if_ready(self):
        self.start_button.setEnabled(bool(self.reference_pdf_path and self.target_folder_path))

    def start_analysis(self):
        if not self.reference_pdf_path or not self.target_folder_path:
            QMessageBox.warning(self, "Eksik Seçim", "Lütfen bir Referans CV ve hedef klasör seçin.")
            return

        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results_tree.clear()
        self.btn_open_cv_file.setEnabled(False)
        self.analysis_results = []
        self._clear_cv_selection()

        self.statusBar().showMessage("Analiz başlatılıyor...")
        self.status_label.setText("Durum: Analiz Başlatılıyor...")

        self.worker = NERAnalysisWorker(self.reference_pdf_path, self.target_folder_path, self.settings)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.analysis_completed.connect(self.analysis_finished)
        self.worker.error_occurred.connect(self.analysis_error)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.start()

    def analysis_finished(self, results: List[Dict[str, Any]]):
        self.analysis_results = results
        self.apply_view_preferences()
        self.start_button.setEnabled(True)
        self.statusBar().showMessage("Analiz tamamlandı.")
        self.status_label.setText("Durum: Analiz Tamamlandı")

        if not results:
            QMessageBox.information(self, "Sonuç Yok", "Belirtilen klasörde analiz edilecek geçerli CV bulunamadı veya tüm CV'lerde hata oluştu.")
        else:
            QMessageBox.information(self, "Analiz Tamamlandı", f"{len(results)} adet CV analizi tamamlandı.")

    def analysis_error(self, message: str):
        QMessageBox.critical(self, "Analiz Hatası", message)
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Analiz hatayla sonuçlandı.")
        self.status_label.setText(f"Durum: Hata - {message}")

    def open_selected_cv_file(self):
        self._open_file(self.current_selected_cv_path, "CV dosyası")

    def _open_file(self, path: str, file_type: str):
        """Genel amaçlı dosya açma metodu."""
        if not path:
            QMessageBox.warning(self, "Uyarı", f"Açılacak bir {file_type} seçili değil.")
            return
        if not os.path.exists(path):
            QMessageBox.critical(self, "Hata", f"Dosya bulunamadı: {path}")
            return

        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            self.statusBar().showMessage(f"{file_type.title()} açıldı: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Dosya Açma Hatası", f"{file_type.title()} açılamadı:\n{e}")
            self.statusBar().showMessage(f"{file_type.title()} açılırken hata oluştu.")

    def show_selected_cv_picture(self):
        if not self.current_selected_cv_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir CV seçin.")
            return

        loading_msg = self._show_loading_message("Profil resmi aranıyor ve işleniyor...")
        image_bytes = None
        try:
            image_bytes = extract_profile_picture_from_pdf_enhanced(self.current_selected_cv_path)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Profil resmi çıkarılırken hata oluştu:\n{str(e)}")
        finally:
            loading_msg.close()
            loading_msg.deleteLater()
            QApplication.processEvents()

        if image_bytes:
            self._display_image_dialog(image_bytes)
        else:
            QMessageBox.information(self, "Bilgi", "Bu CV'de profil resmi bulunamadı.")

    def _show_loading_message(self, text: str):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Lütfen Bekleyin")
        msg_box.setText(text)
        msg_box.setStandardButtons(QMessageBox.NoButton)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.show()
        QApplication.processEvents()
        return msg_box

    def _display_image_dialog(self, image_bytes):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Profil Resmi - {os.path.basename(self.current_selected_cv_path)}")
        dialog.setModal(True)
        dialog_layout = QVBoxLayout()

        image_label = QLabel()
        pixmap = QPixmap()
        if not pixmap.loadFromData(image_bytes):
            QMessageBox.warning(self, "Hata", "Resim yüklenemedi.")
            return

        max_size = 600
        scaled_pixmap = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border: 2px solid #cccccc; padding: 10px;")

        info_label = QLabel(f"Boyut: {pixmap.width()}x{pixmap.height()} piksel")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666666; font-size: 12px; margin-top: 5px;")

        button_layout = QHBoxLayout()
        save_button = QPushButton("Resmi Kaydet")
        save_button.clicked.connect(lambda: self.save_profile_image(image_bytes))
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(dialog.close)
        button_layout.addWidget(save_button)
        button_layout.addStretch()
        button_layout.addWidget(close_button)

        dialog_layout.addWidget(image_label)
        dialog_layout.addWidget(info_label)
        dialog_layout.addLayout(button_layout)
        dialog.setLayout(dialog_layout)
        dialog.resize(max(400, scaled_pixmap.width() + 60), max(300, scaled_pixmap.height() + 120))
        dialog.exec_()

    def save_profile_image(self, image_bytes):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Profil Resmini Kaydet",
            f"profil_resmi_{os.path.basename(self.current_selected_cv_path).split('.')[0]}.png",
            "PNG Dosyaları (*.png);;JPEG Dosyaları (*.jpg);;Tüm Dosyalar (*)"
        )
        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    f.write(image_bytes)
                QMessageBox.information(self, "Başarılı", f"Profil resmi kaydedildi:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Kaydetme Hatası", f"Resim kaydedilirken hata oluştu:\n{str(e)}")

    # ----------------- Yardım -----------------
    def show_shortcuts(self):
        shortcuts_text = """
        <h2>Klavye Kısayolları</h2>
        <ul>
            <li><b>Ctrl+0</b>: Referans CV Seç</li>
            <li><b>Ctrl+Shift+0</b>: Hedef Klasör Seç</li>
            <li><b>Ctrl+E</b>: Sonuçları Dışa Aktar</li>
            <li><b>Ctrl+D</b>: Özet Panelini Gizle/Göster</li>
            <li><b>Ctrl+Q</b>: Uygulamadan Çıkış</li>
            <li><b>Ctrl+,</b>: Gelişmiş Ayarlar</li>
            <li><b>Ctrl+Shift+P</b>: Görüntüleme Tercihleri</li>
        </ul>
        """
        QMessageBox.information(self, "Klavye Kısayolları", shortcuts_text)

    def show_about(self):
        about_text = """
        <h2>CV Karşılaştırma Sistemi v1.0</h2>
        <p>Bu uygulama, işverenlerin aday CV'lerini belirli kriterlere göre analiz etmelerine ve karşılaştırmalarına yardımcı olmak için tasarlanmıştır.</p>
        <p>Geliştirici: [Ahmet Emir ÇAMLIBEL / Niltek Yazılım Teknolojileri A.Ş.]</p>
        <p>Sürüm: 5.0</p>
        <p>Yıl: 2025</p>
        """
        QMessageBox.about(self, "Hakkında", about_text)

    # ----------------- Ayarlar kaydet/yükle -----------------
    def open_preferences(self):
        dialog = PreferencesDialog(self.settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.settings = dialog.get_settings()
            self.apply_theme_and_font()
            self.statusBar().showMessage("Ayarlar kaydedildi ve uygulandı.")

    def save_settings_to_file(self, filename='cv_settings.json'):
        settings_to_save = self.settings.copy()
        if 'font' in settings_to_save and isinstance(settings_to_save['font'], QFont):
            qfont_obj = settings_to_save['font']
            settings_to_save['font'] = {
                'family': qfont_obj.family(), 'pointSize': qfont_obj.pointSize(),
                'weight': qfont_obj.weight(), 'italic': qfont_obj.italic(),
                'bold': qfont_obj.bold()
            }
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ayarlar kaydedilemedi: {e}")

    def load_settings_from_file(self, filename='cv_settings.json'):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)

            if 'font' in loaded_settings and isinstance(loaded_settings['font'], dict):
                font_data = loaded_settings['font']
                font = QFont(font_data.get('family', 'Segoe UI'))
                font.setPointSize(font_data.get('pointSize', 10))
                font.setWeight(font_data.get('weight', QFont.Normal))
                font.setItalic(font_data.get('italic', False))
                font.setBold(font_data.get('bold', False))
                loaded_settings['font'] = font

            self.settings.update(loaded_settings)

        except FileNotFoundError:
            print(f"Ayar dosyası bulunamadı: {filename}")
        except Exception as e:
            print(f"Ayarlar yüklenemedi: {e}")

    def closeEvent(self, event):
        self.save_settings_to_file()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Uygulama başlangıcında varsayılan fontu ayarla
    app.setFont(CVComparisonGUI.DEFAULT_SETTINGS['font'])

    window = CVComparisonGUI()
    window.load_settings_from_file()

    window.show()
    sys.exit(app.exec_())