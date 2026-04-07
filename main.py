"""
CV Karşılaştırma Sistemi - Ana Program
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from gui import CVComparisonGUI

def main():
    """Ana program fonksiyonu"""
    app = QApplication(sys.argv)
    
    # Uygulama ayarları
    app.setApplicationName("CV Karşılaştırma Sistemi")
    app.setApplicationVersion("5.0")
    app.setOrganizationName("CV Analyzer")
    
    # Ana pencereyi oluştur ve göster
    window = CVComparisonGUI()
    window.show()
    
    # Uygulamayı çalıştır
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()