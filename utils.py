"""
CV Karşılaştırma Sistemi - Yardımcı Fonksiyonlar
"""

import os
import re
import string
from datetime import datetime
from typing import List, Dict, Any, Optional
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Gerekli NLTK verilerini indirme (uygulama ilk çalıştığında bir kez yapılmalı veya eksikse indirilmeli)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# PDF okuma için
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Word dosyası okuma için
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


def extract_text_from_file(file_path: str) -> str:
    """
    Belirtilen dosya yolundan metin içeriğini çıkarır.
    Desteklenen formatlar: .pdf, .docx, .txt
    """
    text = ""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        if PDF_AVAILABLE:
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text() or ''
            except Exception as e:
                print(f"PDF okuma hatası: {e}")
                raise
        else:
            raise ImportError("PyPDF2 kütüphanesi yüklü değil. PDF dosyaları okunamıyor.")
    elif file_extension == '.docx':
        if DOCX_AVAILABLE:
            try:
                doc = Document(file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + '\n'
            except Exception as e:
                print(f"DOCX okuma hatası: {e}")
                raise
        else:
            raise ImportError("python-docx kütüphanesi yüklü değil. DOCX dosyaları okunamıyor.")
    elif file_extension == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"TXT okuma hatası: {e}")
            raise
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {file_extension}")
    return text

def normalize_text(text: str, stopwords_enabled: bool = True, lemmatization_enabled: bool = False) -> str:
    """
    Metni küçük harfe dönüştürür, noktalama işaretlerini kaldırır,
    gereksiz boşlukları siler ve isteğe bağlı olarak durdurma kelimelerini kaldırır ve lematizasyon yapar.
    """
    text = text.lower()
    text = re.sub(r'[^a-zçğıiöşü\s]', '', text) # Sadece Türkçe karakterleri ve boşlukları bırak

    # Ek boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text, language='turkish') # Türkçe tokenizasyon

    if stopwords_enabled:
        turkish_stopwords = set(stopwords.words('turkish'))
        tokens = [word for word in tokens if word not in turkish_stopwords]

    if lemmatization_enabled:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens] # İngilizce için çalışır, Türkçe için daha gelişmiş bir lemmatizer gerekebilir

    return ' '.join(tokens)

# Geliştirilmiş tarih desenleri
DATE_PATTERN = r'\b(?:' \
               r'(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+\d{4}|' \
               r'\d{1,2}\s+(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+\d{4}|' \
               r'\d{1,2}\.\d{1,2}\.\d{4}|' \
               r'\d{4}|' \
               r'(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+\d{2}|' \
               r'\d{1,2}\s+(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+\d{2}|' \
               r'günümüz|halen|devam ediyor|present|current' \
               r')\b'

# Ay adlarını numaralarına dönüştürmek için bir sözlük
MONTH_MAPPING = {
    'ocak': 1, 'şubat': 2, 'mart': 3, 'nisan': 4, 'mayıs': 5, 'haziran': 6,
    'temmuz': 7, 'ağustos': 8, 'eylül': 9, 'ekim': 10, 'kasım': 11, 'aralık': 12
}

def parse_date(date_str: str) -> Optional[datetime]:
    """
    Metin olarak verilen tarihi datetime objesine dönüştürür.
    "Günümüz" veya "Halen" gibi ifadeleri de işler.
    """
    date_str = date_str.lower().strip()

    if date_str in ['günümüz', 'halen', 'devam ediyor', 'present', 'current']:
        return datetime.now()

    # YYYY-MM-DD (örneğin 2023-01-15)
    match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    # DD.MM.YYYY (örneğin 15.01.2023)
    match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', date_str)
    if match:
        try:
            return datetime(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        except ValueError:
            pass

    # Ay Adı YYYY (örneğin Ocak 2023)
    match = re.search(r'(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+(\d{4})', date_str)
    if match:
        try:
            month = MONTH_MAPPING.get(match.group(1))
            year = int(match.group(2))
            if month:
                return datetime(year, month, 1)
        except ValueError:
            pass

    # YYYY (örneğin 2023)
    match = re.search(r'\b(\d{4})\b', date_str)
    if match and len(date_str.strip()) == 4: # Sadece yıl içeren stringler için
        try:
            return datetime(int(match.group(1)), 1, 1)
        except ValueError:
            pass

    # DD Ay Adı YYYY (örneğin 15 Ocak 2023)
    match = re.search(r'(\d{1,2})\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+(\d{4})', date_str)
    if match:
        try:
            day = int(match.group(1))
            month = MONTH_MAPPING.get(match.group(2))
            year = int(match.group(3))
            if month:
                return datetime(year, month, day)
        except ValueError:
            pass
            
    # Ay Adı YY (örneğin Ocak 23)
    match = re.search(r'(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+(\d{2})', date_str)
    if match:
        try:
            month = MONTH_MAPPING.get(match.group(1))
            year_short = int(match.group(2))
            # 2000'li veya 1900'lü yılları tahmin et
            if year_short >= 0 and year_short <= datetime.now().year % 100 + 5: # Şimdiki yılın 2 basamağından küçük veya eşitse 2000'ler, aksi takdirde 1900'ler
                year = 2000 + year_short
            else:
                year = 1900 + year_short
            if month:
                return datetime(year, month, 1)
        except ValueError:
            pass

    # DD Ay Adı YY (örneğin 15 Ocak 23)
    match = re.search(r'(\d{1,2})\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+(\d{2})', date_str)
    if match:
        try:
            day = int(match.group(1))
            month = MONTH_MAPPING.get(match.group(2))
            year_short = int(match.group(3))
            if year_short >= 0 and year_short <= datetime.now().year % 100 + 5:
                year = 2000 + year_short
            else:
                year = 1900 + year_short
            if month:
                return datetime(year, month, day)
        except ValueError:
            pass

    return None

def calculate_months_between_dates(start_date: datetime, end_date: datetime) -> int:
    """
    İki tarih arasındaki ay farkını hesaplar.
    """
    if not start_date or not end_date:
        return 0

    # Bitiş tarihi başlangıç tarihinden küçükse veya eşitse 0 döndür
    if end_date <= start_date:
        return 0

    # Ay farkını hesapla
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    # Gün farkını da dikkate alarak daha hassas bir hesaplama yapabiliriz.
    # Eğer bitiş ayının günü, başlangıç ayının gününden küçükse, bir ay eksilt.
    if end_date.day < start_date.day:
        total_months -= 1

    return max(0, total_months) # Negatif değer olmaması için 0'dan büyük olanı al

def format_total_months_to_years_months(total_months: int) -> str:
    """
    Toplam ay sayısını "Yıl X Ay Y" formatına dönüştürür.
    """
    if total_months < 0:
        return "Geçersiz Süre"
    years = total_months // 12
    remaining_months = total_months % 12
    
    parts = []
    if years > 0:
        parts.append(f"{years} Yıl")
    if remaining_months > 0:
        parts.append(f"{remaining_months} Ay")
    
    if not parts:
        return "0 Ay"
    
    return " ".join(parts)