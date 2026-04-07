"""
CV Karşılaştırma Sistemi - Optimize Edilmiş Analiz Motoru
"""

import os
import json
import re
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import fitz
import tempfile
import io
from PIL import Image

# Gerekli kütüphaneler için güvenli import
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    import spacy
    nlp = spacy.blank("en")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# utils.py'den import için fallback fonksiyonlar
try:
    from utils import extract_text_from_file, normalize_text, DATE_PATTERN, parse_date, calculate_months_between_dates, format_total_months_to_years_months
except ImportError:
    def extract_text_from_file(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def normalize_text(text: str, stopwords_enabled: bool = True, lemmatization_enabled: bool = False) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zçğıiöşü\s]', '', text)
        return text

    DATE_PATTERN = r'\b(?:' \
                   r'(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}|' \
                   r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|' \
                   r'\d{4}|' \
                   r'current|present|devam ediyor' \
                   r')\b'
    
    def parse_date(date_str: str) -> Optional[datetime]: 
        return None
    
    def calculate_months_between_dates(start_date: datetime, end_date: datetime) -> int: 
        return 0
    
    def format_total_months_to_years_months(total_months: int) -> str: 
        return f"{total_months // 12} yıl {total_months % 12} ay"


def extract_text_from_pdf(file_path: str) -> str:
    """PDF dosyasından metin çıkarımı"""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_profile_picture_from_pdf(file_path: str) -> Optional[bytes]:
    """PDF'den profil resmi çıkarır"""
    try:
        doc = fitz.open(file_path)
        first_page = doc[0]
        image_list = first_page.get_images(full=True)

        if image_list:
            xref = image_list[0][0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            if len(image_bytes) > 5000:
                img = Image.open(io.BytesIO(image_bytes))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                doc.close()
                return img_byte_arr.getvalue()
        
        doc.close()
        return None
    except Exception as e:
        print(f"Resim çıkarma hatası: {e}")
        return None


class CVAnalyzer:
    """CV analiz ve karşılaştırma sınıfı"""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Analyzer'ı başlat"""
        self.settings = settings or self._get_default_settings()
        self.nlp = nlp
        
        # Semantic model yükleme
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('paraphrase-mpnet-base-v2')
            except Exception as e:
                print(f"[WARN] Semantic model yüklenemedi: {e}")

        # Translator başlatma
        if TRANSLATOR_AVAILABLE:
            try:
                self.translator = GoogleTranslator(source='auto', target='tr')
                self.translate_available = True
            except Exception:
                self.translator = None
                self.translate_available = False
        else:
            self.translator = None
            self.translate_available = False

        # Regex desenleri
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')

        # Bölüm başlıkları
        self.section_headers = {
            'personal_info': [r'kişisel bilgiler', r'iletişim', r'contact', r'personal'],
            'education': [r'eğitim', r'education', r'akademik'],
            'experience': [r'iş deneyimi', r'deneyim', r'experience', r'employment'],
            'skills': [r'beceriler', r'skills', r'yetkinlikler'],
            'certifications': [r'sertifikalar', r'certifications']
        }
        
        # Anahtar kelime listeleri
        self.degree_keywords = [
            r'\b(?:lisans|yüksek lisans|doktora|ön lisans|master|phd|bachelor|degree|diploma)\b'
        ]
        self.department_keywords = [
            r'\b(?:mühendislik|bilgisayar|yazılım|işletme|tıp|hukuk|computer science|engineering)\b'
        ]
        self.gpa_patterns = [
            r'not ortalaması:?\s*(\d(?:\.\d{1,2})?)',
            r'\b(?:gpa|cgn):?\s*(\d(?:\.\d{1,2})?)',
            r'ort:?\s*(\d(?:\.\d{1,2})?)\b'
        ]
        
    def _get_default_settings(self) -> Dict[str, Any]:
        """Varsayılan ayarları döndürür"""
        return {
            'minimum_similarity_threshold': 50,
            'section_weights': {
                'personal_info': 0.1,
                'education': 0.2,
                'experience': 0.4,
                'skills': 0.2,
                'certifications': 0.1
            },
            'stopwords_enabled': True,
            'lemmatization_enabled': False,
        }

    def extract_cv_data(self, cv_path: str) -> Dict[str, Any]:
        """CV dosyasından veri çıkarımı yapar"""
        try:
            text = extract_text_from_pdf(cv_path)
            
            if self._is_special_format(text):
                return self._extract_cv_data_special_format(text)
            
            return self._extract_cv_data_normal_format(text)

        except Exception as e:
            raise Exception(f"CV verisi çıkarılırken hata oluştu: {str(e)}")

    def _is_special_format(self, text: str) -> bool:
        """Özel format kontrolü"""
        special_indicators = [
            r"İsim Soyisim:",
            r"E_posta\s*[=:]",
            r"Tel No\s*[=:]",
            r"Eğitim Bilgisi:",
            r"Ortak Beceriler:"
        ]
        
        matches = sum(1 for pattern in special_indicators 
                     if re.search(pattern, text, re.I))
        return matches >= 3

    def _extract_cv_data_special_format(self, text: str) -> Dict[str, Any]:
        """Özel format için veri çıkarımı"""
        data = {
            'full_name': 'Bilinmiyor',
            'email': None,
            'phone': None,
            'personal_info': {},
            'education': [],
            'experience': [],
            'skills': [],
            'certifications': [],
            'entities': {},
            'raw_text': text,
            'normalized_text': normalize_text(text, 
                                             self.settings.get('stopwords_enabled', True),
                                             self.settings.get('lemmatization_enabled', False))
        }

        # İsim çıkarımı
        name_match = re.search(r"İsim Soyisim:\s*(.+)", text, re.I)
        if name_match:
            data['full_name'] = name_match.group(1).strip()
        
        # E-posta çıkarımı
        email_match = re.search(r"E_posta(?:=|\s*[:])\s*(\S+@\S+)", text, re.I)
        if email_match:
            data['email'] = email_match.group(1).strip()

        # Telefon çıkarımı
        phone_match = re.search(r"Tel No(?:=|\s*[:])\s*(\d+)", text, re.I)
        if phone_match:
            data['phone'] = phone_match.group(1).strip()

        # Eğitim çıkarımı
        education_match = re.search(r"Eğitim Bilgisi:\s*(.+?)(?=\n(?:Deneyim|Ortak Beceriler|Sertifikalar|$))", text, re.S | re.I)
        if education_match:
            edu_text = education_match.group(1).strip()
            edu_parts = [part.strip() for part in edu_text.split('/')]
            if len(edu_parts) >= 2:
                data['education'].append({
                    'university': [edu_parts[0]] if edu_parts[0] else [],
                    'department': [edu_parts[1]] if len(edu_parts) > 1 and edu_parts[1] else [],
                    'gpa': edu_parts[2].replace('ORT:', '').strip() if len(edu_parts) > 2 else None,
                    'degree': [],
                    'dates_raw': []
                })
        
        # Deneyim çıkarımı
        experience_match = re.search(r"Deneyim:\s*(.+?)(?=\n(?:Ortak Beceriler|Sertifikalar|$))", text, re.S | re.I)
        if experience_match:
            exp_text = experience_match.group(1).strip()
            exp_lines = [line.strip() for line in exp_text.split('\n') if line.strip()]
            for line in exp_lines:
                company_match = re.match(r"(.+?)\s*\((.+?)\)", line)
                if company_match:
                    data['experience'].append({
                        'company': [company_match.group(1).strip()],
                        'duration_formatted': company_match.group(2).strip(),
                        'title': []
                    })
                else:
                    data['experience'].append({
                        'company': [line],
                        'duration_formatted': 'Belirtilmemiş',
                        'title': []
                    })
        
        # Beceriler çıkarımı
        skills_match = re.search(r"Ortak Beceriler:\s*(.+?)(?=\n(?:Sertifikalar|$))", text, re.S | re.I)
        if skills_match:
            skills_text = skills_match.group(1).strip()
            data['skills'] = [s.strip() for s in skills_text.split(',') if s.strip()]

        # Sertifikalar çıkarımı
        certificates_match = re.search(r"Sertifikalar:\s*(.+?)$", text, re.S | re.I)
        if certificates_match:
            cert_text = certificates_match.group(1).strip()
            data['certifications'] = [c.strip() for c in cert_text.split('\n') if c.strip()]

        # Personal info güncelle
        data['personal_info'] = {
            'full_name': data['full_name'],
            'email': data['email'],
            'phone': data['phone']
        }

        data['entities'] = self._extract_entities_with_ner(text)
        return data

    def _extract_cv_data_normal_format(self, text: str) -> Dict[str, Any]:
        """Normal format için veri çıkarımı"""
        sections = self._split_into_sections(text)
        personal_info = self._extract_personal_info_comprehensive(text)

        skills = self._extract_comprehensive_skills(sections.get('skills', '')) if 'skills' in sections else []
        education = self._parse_education_with_ner(sections.get('education', '')) if 'education' in sections else []
        experience = self._parse_experience_with_ner(sections.get('experience', '')) if 'experience' in sections else []
        certifications = self._extract_certifications(sections.get('certifications', '')) if 'certifications' in sections else []

        entities = self._extract_entities_with_ner(text)

        return {
            'full_name': personal_info.get('full_name', 'Bilinmiyor'),
            'email': personal_info.get('email'),
            'phone': personal_info.get('phone'),
            'personal_info': personal_info,
            'education': education,
            'experience': experience,
            'skills': skills,
            'certifications': certifications,
            'entities': entities,
            'raw_text': text,
            'normalized_text': normalize_text(text, 
                                             self.settings.get('stopwords_enabled', True),
                                             self.settings.get('lemmatization_enabled', False))
        }

    def _translate_to_turkish(self, text: str) -> str:
        """Metni Türkçeye çevirir"""
        if not self.translate_available or not text.strip():
            return text
        
        try:
            simple_translations = {
                'bachelor': 'lisans',
                'master': 'yüksek lisans',
                'phd': 'doktora',
                'university': 'üniversite',
                'software': 'yazılım',
                'developer': 'geliştirici',
                'engineer': 'mühendis',
                'manager': 'yönetici'
            }
            
            if len(text.split()) < 5:
                translated_simple = simple_translations.get(text.lower(), text)
                if translated_simple != text:
                    return translated_simple
            
            translated = self.translator.translate(text)
            return translated if translated else text
        
        except Exception:
            return text

    def _extract_entities_with_ner(self, text: str) -> Dict[str, List[str]]:
        """NER ile varlık çıkarımı"""
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'DATE': [],
            'DEGREE': [],
            'DEPARTMENT': []
        }

        # spaCy etiketleri
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['PERSON'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['ORG'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['DATE'].append(ent.text)

        # Kural tabanlı çıkarım
        for pattern in self.degree_keywords:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                degree_text = match.group(0)
                translated_degree = self._translate_to_turkish(degree_text)
                entities['DEGREE'].append(translated_degree)
        
        for pattern in self.department_keywords:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dept_text = match.group(0)
                translated_dept = self._translate_to_turkish(dept_text)
                entities['DEPARTMENT'].append(translated_dept)

        return {k: list(set(v)) for k, v in entities.items()}

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Metni bölümlere ayırır"""
        sections = {'raw_text': text}
        text_lower = text.lower()
        
        section_indices = []
        for section_key, patterns in self.section_headers.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    section_indices.append((match.start(), section_key, match.group(0)))

        section_indices.sort()

        for i, (start_idx, section_key, header_text) in enumerate(section_indices):
            end_idx = section_indices[i+1][0] if i + 1 < len(section_indices) else len(text)
            section_content = text[start_idx:end_idx].strip()

            if section_key in sections:
                sections[section_key] += "\n" + section_content
            else:
                sections[section_key] = section_content
        
        return sections
    
    def _extract_personal_info_comprehensive(self, text: str) -> Dict[str, str]:
        """Kişisel bilgi çıkarımı"""
        personal_info = {
            'full_name': 'Bilinmiyor',
            'email': None,
            'phone': None
        }

        first_lines = "\n".join(text.split('\n')[:20])

        # Email
        email_match = self.email_pattern.search(first_lines)
        if email_match:
            personal_info['email'] = email_match.group(0)

        # Telefon
        phone_patterns = [
            r'(?:\+90|0)?[\s-]?\(?5\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}',
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{10}'
        ]

        for pattern in phone_patterns:
            phone_match = re.search(pattern, first_lines)
            if phone_match:
                personal_info['phone'] = phone_match.group(0)
                break

        # İsim çıkarımı
        doc = self.nlp(first_lines)
        potential_names = []

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                potential_names.append(ent.text)

        first_line = text.split('\n')[0] if text.split('\n') else ""
        name_candidates = re.findall(r'\b[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*\b', first_line)
        potential_names.extend(name_candidates)

        if potential_names:
            valid_names = [name for name in potential_names
                           if not re.search(r'@|\.com|\d{3,}', name) and len(name.split()) >= 2]
            if valid_names:
                personal_info['full_name'] = max(valid_names, key=len)
            elif potential_names:
                personal_info['full_name'] = potential_names[0]

        return personal_info
    
    def _extract_comprehensive_skills(self, text: str) -> List[str]:
        """Beceri çıkarımı"""
        skills = set()

        tech_skills = [
            'Python', 'Java', 'JavaScript', 'C++', 'C#', 'PHP', 'Ruby', 'Go',
            'React', 'Angular', 'Vue.js', 'Django', 'Flask', 'Spring',
            'MySQL', 'PostgreSQL', 'MongoDB', 'Oracle', 'SQL',
            'Git', 'Docker', 'AWS', 'Azure', 'Linux'
        ]

        text_lower = text.lower()
        for skill in tech_skills:
            if skill.lower() in text_lower:
                skills.add(skill)

        skill_patterns = [
            r'(?:skills?|beceriler?):?\s*([^.!?\n]+)',
            r'(?:programming|programlama):?\s*([^.!?\n]+)',
            r'(?:technologies|teknolojiler):?\s*([^.!?\n]+)'
        ]

        for pattern in skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                skill_text = match.group(1)
                individual_skills = re.split(r'[,;•\-\n\t\r]+', skill_text)
                for skill in individual_skills:
                    clean_skill = skill.strip()
                    if len(clean_skill) > 1 and not re.search(r'\d{4}|@|\.com', clean_skill):
                        skills.add(clean_skill)

        return list(skills)

    def _parse_education_with_ner(self, section_text: str) -> List[Dict[str, Any]]:
        """Eğitim bilgisi çıkarımı"""
        educations = []
        education_entries = re.split(r'\n\s*\n|\d{4}', section_text.strip())
        
        for entry_text in education_entries:
            if not entry_text.strip() or len(entry_text.strip()) < 10:
                continue

            entry_doc = self.nlp(entry_text)
            found_degree = []
            found_university = []
            found_department = []
            gpa = None

            for ent in entry_doc.ents:
                if ent.label_ == 'ORG':
                    found_university.append(ent.text)
                
            for gpa_pattern in self.gpa_patterns:
                gpa_match = re.search(gpa_pattern, entry_text, re.IGNORECASE)
                if gpa_match and len(gpa_match.groups()) > 0:
                    gpa = gpa_match.group(1)
                    break

            for pattern in self.degree_keywords:
                for match in re.finditer(pattern, entry_text, re.IGNORECASE):
                    found_degree.append(match.group(0))

            for pattern in self.department_keywords:
                for match in re.finditer(pattern, entry_text, re.IGNORECASE):
                    found_department.append(match.group(0))

            if found_degree or found_university or found_department or gpa:
                educations.append({
                    'degree': list(set(found_degree)),
                    'university': list(set(found_university)),
                    'department': list(set(found_department)),
                    'gpa': gpa
                })
        return educations

    def _parse_experience_with_ner(self, section_text: str) -> List[Dict[str, Any]]:
        """Deneyim bilgisi çıkarımı"""
        experiences = []
        experience_entries = re.split(r'\n\s*\n', section_text.strip())

        for entry in experience_entries:
            if not entry.strip() or len(entry.strip()) < 10:
                continue

            entry_doc = self.nlp(entry)
            found_company = []

            for ent in entry_doc.ents:
                if ent.label_ == 'ORG':
                    found_company.append(ent.text)

            if found_company:
                experiences.append({
                    'company': list(set(found_company)),
                    'title': []
                })
        return experiences

    def _extract_certifications(self, text: str) -> List[str]:
        """Sertifika çıkarımı"""
        certifications = []
        cert_candidates = re.split(r'[,;\n\t\r-•]', text)
        
        for cert_candidate in cert_candidates:
            clean_cert = cert_candidate.strip()
            if len(clean_cert) > 3 and clean_cert.lower() not in ['ve', 'ile', 'and', 'with', '&']:
                certifications.append(clean_cert)
        
        return list(set(certifications))

    def _calculate_total_experience_years(self, experiences: List[Dict[str, Any]]) -> float:
        """Toplam deneyim hesaplama"""
        total_months = 0
        for exp in experiences:
            if exp.get('duration_months'):
                total_months += exp['duration_months']
        return round(total_months / 12.0, 1)

    def _calculate_overall_similarity(self, matches: Dict[str, Any]) -> float:
        """Genel benzerlik hesaplama"""
        section_weights = self.settings.get('section_weights', {})
        total_score = 0.0
        total_weight = 0.0
        
        all_scores_perfect = True
        
        for section, weight in section_weights.items():
            match_key = f"{section}_match"
            if match_key in matches:
                score = matches[match_key].get('score', 0.0)
                total_score += score * weight
                total_weight += weight
                
                if score < 99.9:
                    all_scores_perfect = False
        
        if all_scores_perfect and total_weight > 0:
            return 100.0
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def _format_analysis_summary(self, cv_data: Dict[str, Any], overall_similarity: float, matches: Dict[str, Any], is_suitable: bool) -> str:
        """Analiz özeti formatla"""
        summary_parts = []
        
        personal_info = cv_data.get('personal_info', {})
        name = personal_info.get('full_name', cv_data.get('full_name', 'Bilinmiyor'))
        email = personal_info.get('email', cv_data.get('email'))
        phone = personal_info.get('phone', cv_data.get('phone'))
        
        summary_parts.append(f"**👤 Ad Soyad:** {name}")
        
        if email:
            summary_parts.append(f"**📧 E-posta:** {email}")
        if phone:
            summary_parts.append(f"**📱 Telefon:** {phone}")

        summary_parts.append(f"\n**📊 Genel Benzerlik Skoru:** %{overall_similarity:.2f}")
        if is_suitable:
            summary_parts.append("**✅ Durum: Uygun**")
        else:
            min_threshold = self.settings.get('minimum_similarity_threshold', 50)
            summary_parts.append(f"**❌ Durum: Uygun Değil** (Eşik: %{min_threshold})")
        
        return '\n'.join(summary_parts)

    def _calculate_section_similarity(self, ref_section, target_section):
        """Bölüm benzerliği hesaplama"""
        if isinstance(ref_section, list):
            ref_items = [str(x).strip().lower() for x in ref_section if str(x).strip()]
            ref_items.sort()
            ref_text = " ".join(ref_items).strip()
        else:
            ref_text = str(ref_section or "").strip().lower()

        if isinstance(target_section, list):
            target_items = [str(x).strip().lower() for x in target_section if str(x).strip()]
            target_items.sort()
            target_text = " ".join(target_items).strip()
        else:
            target_text = str(target_section or "").strip().lower()

        if (not ref_text) and (not target_text):
            return {"score": 100.0, "intersection": []}

        if not ref_text or not target_text:
            return {"score": 0.0, "intersection": []}

        if ref_text == target_text:
            intersection = list(set(ref_text.split()))
            return {"score": 100.0, "intersection": intersection}

        # Token bazlı Jaccard similarity
        ref_tokens = set(re.findall(r"\w+", ref_text, flags=re.UNICODE))
        target_tokens = set(re.findall(r"\w+", target_text, flags=re.UNICODE))
        intersection = ref_tokens.intersection(target_tokens)
        union = ref_tokens.union(target_tokens)
        jaccard_score = (len(intersection) / len(union)) * 100 if union else 0.0

        # Semantic similarity
        semantic_score = None
        if self.semantic_model is not None:
            try:
                embeddings = self.semantic_model.encode([ref_text, target_text], convert_to_tensor=True)
                semantic_score = util.cos_sim(embeddings[0], embeddings[1]).item() * 100
            except Exception:
                semantic_score = None

        if semantic_score is None:
            hybrid_score = jaccard_score
        else:
            if jaccard_score >= 99.9:
                hybrid_score = 100.0
            else:
                hybrid_score = (0.7 * semantic_score) + (0.3 * jaccard_score)

        return {"score": float(hybrid_score), "intersection": list(intersection)}

    def _get_section_content_for_similarity(self, cv_data: Dict[str, Any], section_key: str) -> List[str]:
        """Benzerlik hesaplaması için bölüm içeriği"""
        def normalize_string(s):
            return str(s).strip().lower() if s else ""
        
        if section_key == 'education':
            education_strings = []
            for edu in cv_data.get('education', []):
                degree_str = normalize_string(self._get_value_as_string(edu, 'degree'))
                university_str = normalize_string(self._get_value_as_string(edu, 'university'))
                department_str = normalize_string(self._get_value_as_string(edu, 'department'))
                gpa_str = normalize_string(edu.get('gpa', ''))
                
                edu_combined = f"{degree_str} {university_str} {department_str} {gpa_str}".strip()
                if edu_combined:
                    education_strings.append(edu_combined)
            return education_strings
            
        elif section_key == 'experience':
            experience_strings = []
            for exp in cv_data.get('experience', []):
                company_str = normalize_string(self._get_value_as_string(exp, 'company'))
                title_str = normalize_string(self._get_value_as_string(exp, 'title'))
                
                exp_combined = f"{title_str} {company_str}".strip()
                if exp_combined:
                    experience_strings.append(exp_combined)
            return experience_strings
            
        elif section_key == 'skills':
            skills = cv_data.get('skills', [])
            return [normalize_string(skill) for skill in skills if normalize_string(skill)]
            
        elif section_key == 'personal_info':
            info_list = []
            personal_info = cv_data.get('personal_info', {})
            
            if isinstance(personal_info, dict):
                if personal_info.get('full_name'):
                    info_list.append(normalize_string(personal_info['full_name']))
                if personal_info.get('email'):
                    info_list.append(normalize_string(personal_info['email']))
                if personal_info.get('phone'):
                    info_list.append(normalize_string(personal_info['phone']))
            else:
                if cv_data.get('full_name'):
                    info_list.append(normalize_string(cv_data['full_name']))
                if cv_data.get('email'):
                    info_list.append(normalize_string(cv_data['email']))
                    
            return [info for info in info_list if info]
            
        elif section_key == 'certifications':
            certs = cv_data.get('certifications', [])
            return [normalize_string(cert) for cert in certs if normalize_string(cert)]
            
        return []

    def _get_value_as_string(self, data_dict: Dict[str, Any], key: str, default_value: Any = '') -> str:
        """Sözlükten değeri alır ve string olarak döndürür"""
        value = data_dict.get(key, default_value)
        if isinstance(value, list):
            return " ".join(str(v) for v in value if v)
        return str(value) if value else ""

    def analyze_cv(self, reference_cv_data: Dict[str, Any], target_cv_path: str) -> Dict[str, Any]:
        """Tek bir hedef CV'yi referans CV ile karşılaştırır"""
        if not os.path.exists(target_cv_path):
            return {
                'file_name': os.path.basename(target_cv_path),
                'person_name': 'Dosya Bulunamadı',
                'overall_similarity': 0.0,
                'text_similarity': 0.0,
                'matches': {},
                'cv_data': {'raw_text': f"Hata: {target_cv_path} dosyası bulunamadı."},
                'formatted_summary': f"Hata: {target_cv_path} dosyası bulunamadı.",
                'is_suitable': False 
            }

        try:
            target_cv_data = self.extract_cv_data(target_cv_path)

            # Aynı dosya kontrolü
            reference_file_name = getattr(self, '_reference_file_name', None)
            target_file_name = os.path.basename(target_cv_path)
            
            if reference_file_name and reference_file_name == target_file_name:
                matches = {
                    'education_match': {"score": 100.0, "intersection": []},
                    'experience_match': {"score": 100.0, "intersection": []},
                    'skills_match': {"score": 100.0, "intersection": []},
                    'certifications_match': {"score": 100.0, "intersection": []}
                }
                overall_similarity = 100.0
                text_similarity = 100.0
            else:
                matches = {
                    'education_match': self._calculate_section_similarity(
                        self._get_section_content_for_similarity(reference_cv_data, 'education'),
                        self._get_section_content_for_similarity(target_cv_data, 'education')
                    ),
                    'experience_match': self._calculate_section_similarity(
                        self._get_section_content_for_similarity(reference_cv_data, 'experience'),
                        self._get_section_content_for_similarity(target_cv_data, 'experience')
                    ),
                    'skills_match': self._calculate_section_similarity(
                        self._get_section_content_for_similarity(reference_cv_data, 'skills'),
                        self._get_section_content_for_similarity(target_cv_data, 'skills')
                    ),
                    'certifications_match': self._calculate_section_similarity(
                        self._get_section_content_for_similarity(reference_cv_data, 'certifications'),
                        self._get_section_content_for_similarity(target_cv_data, 'certifications')
                    )
                }

                overall_similarity = self._calculate_overall_similarity(matches)

                # Metin benzerliği
                ref_normalized = reference_cv_data.get('normalized_text', '').strip().lower()
                target_normalized = target_cv_data.get('normalized_text', '').strip().lower()
                
                if ref_normalized and target_normalized and ref_normalized == target_normalized:
                    text_similarity = 100.0
                else:
                    text_similarity_result = self._calculate_section_similarity(
                        [ref_normalized],
                        [target_normalized]
                    )
                    text_similarity = text_similarity_result.get('score', 0.0)

            minimum_threshold = self.settings.get('minimum_similarity_threshold', 50)
            is_suitable = overall_similarity >= minimum_threshold

            formatted_cv_summary = self._format_analysis_summary(target_cv_data, overall_similarity, matches, is_suitable)

            return {
                'file_name': os.path.basename(target_cv_path),
                'person_name': target_cv_data.get('full_name', 'Bilinmiyor'),
                'overall_similarity': overall_similarity,
                'text_similarity': text_similarity,
                'matches': matches,
                'cv_data': target_cv_data,
                'formatted_summary': formatted_cv_summary,
                'is_suitable': is_suitable
            }

        except Exception as e:
            return {
                'file_name': os.path.basename(target_cv_path),
                'person_name': 'Analiz Hatası',
                'overall_similarity': 0.0,
                'text_similarity': 0.0,
                'matches': {},
                'cv_data': {'error': str(e)},
                'formatted_summary': f"Analiz hatası: {str(e)}",
                'is_suitable': False 
            }

    def analyze_cv_folder(self, reference_pdf_path: str, target_folder_path: str) -> List[Dict[str, Any]]:
        """Referans CV'yi hedef klasördeki tüm CV'lerle karşılaştır"""
        results = []
        
        self._reference_file_name = os.path.basename(reference_pdf_path)
        
        try:
            reference_cv_data = self.extract_cv_data(reference_pdf_path)
        except Exception as e:
            raise Exception(f"Referans CV ({os.path.basename(reference_pdf_path)}) okunurken hata oluştu: {e}")

        cv_files = [os.path.join(target_folder_path, f)
                    for f in os.listdir(target_folder_path)
                    if f.lower().endswith(('.pdf', '.docx', '.txt'))]

        for cv_file in cv_files:
            try:
                analysis_result = self.analyze_cv(reference_cv_data, cv_file)
                results.append(analysis_result)
            except Exception as e:
                print(f"Uyarı: {os.path.basename(cv_file)} analiz edilirken hata oluştu: {e}")
                results.append({
                    'file_name': os.path.basename(cv_file),
                    'person_name': 'Hata Oluştu',
                    'overall_similarity': 0.0,
                    'text_similarity': 0.0,
                    'matches': {},
                    'cv_data': {'raw_text': f"Analiz hatası: {e}"},
                    'formatted_summary': f"Analiz hatası: {e}",
                    'is_suitable': False
                })
        return results


def run_sample_analysis():
    """Örnek CV metni ile test"""
    sample_cv_text = """
    İsim Soyisim: Ahmet Emir ÇAMLIBEL
    E_posta = ahmetcamlibel@gmail.com
    Tel No: 5354103080

    Eğitim Bilgisi:
    Fırat Üniversitesi / Adli Bilişim Mühendisliği / ORT: 2.64

    Deneyim:
    Niltek Yazılım Teknolojileri A.Ş. (Nisan 2024 - Ekim 2024)
    Fırat Teknokent (Kasım 2024 - Mart 2025)

    Ortak Beceriler: React, Python, JS, HTML, CSS, JavaScript, SQL, Django

    Sertifikalar:
    Temel Network 101
    Temel Network 201
    Siber Güvenlik Başarı Sertifikası
    """

    analyzer = CVAnalyzer()
    
    try:
        cv_data = analyzer._extract_cv_data_special_format(sample_cv_text)
        
        output_lines = []
        
        name = cv_data.get('full_name', 'Bilinmiyor')
        if name and name != 'Bilinmiyor':
            output_lines.append(f"İsim Soyisim: {name}")
            
            email = cv_data.get('email')
            phone = cv_data.get('phone')
            
            if email and email != 'Bilinmiyor':
                output_lines.append(f"E_posta = {email}")
            if phone and phone != 'Bilinmiyor':
                output_lines.append(f"Tel No: {phone}")
        
        education_info = cv_data.get('education', [])
        if education_info:
            output_lines.append("")
            output_lines.append("Eğitim Bilgisi:")
            for edu in education_info:
                university = edu.get('university', [])
                department = edu.get('department', [])
                gpa = edu.get('gpa', '')
                
                edu_line = []
                if university: 
                    edu_line.append(university[0] if isinstance(university, list) else str(university))
                if department: 
                    edu_line.append(department[0] if isinstance(department, list) else str(department))
                if gpa: 
                    edu_line.append(f"ORT: {gpa}")
                
                if edu_line:
                    output_lines.append(" / ".join(edu_line))
        
        experience_info = cv_data.get('experience', [])
        if experience_info:
            output_lines.append("")
            output_lines.append("Deneyim:")
            for exp in experience_info:
                company = exp.get('company', [])
                duration = exp.get('duration_formatted', '')
                
                if company:
                    company_name = company[0] if isinstance(company, list) else str(company)
                    exp_line = company_name
                    if duration and duration != 'Belirtilmemiş':
                        exp_line += f" ({duration})"
                    output_lines.append(exp_line)
        
        skills = cv_data.get('skills', [])
        if skills:
            valid_skills = [skill for skill in skills if skill.strip() and len(skill.strip()) > 1]
            if valid_skills:
                output_lines.append("")
                output_lines.append(f"Ortak Beceriler: {', '.join(valid_skills)}")
        
        certificates = cv_data.get('certifications', [])
        if certificates:
            valid_certificates = [cert for cert in certificates if cert.strip() and len(cert.strip()) > 2]
            if valid_certificates:
                output_lines.append("")
                output_lines.append("Sertifikalar:")
                for cert in valid_certificates:
                    output_lines.append(cert)
        
        for line in output_lines:
            print(line)
    
    except Exception as e:
        print(f"Analiz sırasında hata oluştu: {e}")


if __name__ == "__main__":
    run_sample_analysis()