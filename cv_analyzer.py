import os
import fitz
from tkinter import filedialog, messagebox, ttk, Tk
from sentence_transformers import SentenceTransformer, util
import unicodedata
import re
from shutil import copy2
import threading
from datetime import datetime
import gc #Bellek temizleme için
from time import time #Zaman kontrolü için
import torch

model = SentenceTransformer('paraphrase-mpnet-base-v2')
print("Model başarıyla yüklendi.")

def pdf_to_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text =""
        max_pages = min(3, len(doc))

        for page_num in range(max_pages):
            page_text = doc[page_num].get_text()
            text += page_text

            #Metin çok uzunsa kısalt
            if len(text) > 50000:
                 break
            
        doc.close()
        return text
    except Exception as e:
        print(f"❌ Hata: {pdf_path} okunamadı → {e}")
        return ""

#Ön İşleme
def clean_text(text):
    if not text:
         return ""
    
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\s+', ' ',text)
    return text[:10000].strip()


# cv_analyzer.py dosyasına eklenecek yeni fonksiyon
def analyze_cv_folder(prompt_embedding, threshold, folder_path):
    all_results = []
    
    cv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.docx', '.txt'))]

    if not cv_files:
        return []

    for cv_path in cv_files:
        try:
            cv_text = pdf_to_text(cv_path)
            if not cv_text:
                continue

            cleaned_cv_text = clean_text(cv_text)
            cv_embedding = model.encode(cleaned_cv_text, convert_to_tensor=True)
            
            similarity = util.cos_sim(prompt_embedding, cv_embedding).item() * 100
            
            status = "Uygun" if similarity >= threshold else "Uygun Değil"
            
            all_results.append((os.path.basename(cv_path), similarity, status))

        except Exception as e:
            # Hata durumunda boş sonuç ekle ve devam et
            all_results.append((os.path.basename(cv_path), 0.0, "Analiz Hatası"))
            print(f"❌ Analiz hatası ({os.path.basename(cv_path)}): {e}")

    return all_results
     

if __name__ == "__main__":
     print("=" * 60)
     print("🔍 CV ANALİZ SİSTEMİ")
     print("=" * 60)


     #Prompt 
     prompt = input("📝 İş tanımını girin (örnek: Python bilen, 3 yıl deneyimli):\n> ").strip()
     print(f"🔍 DEBUG: Girilen prompt: '{prompt}'")
     print(f"🔍 DEBUG: Prompt uzunluğu: {len(prompt)}")

     print("\n📂 Lütfen PDF CV'lerin bulunduğu klasörü seçin...")
     root = Tk()
     root.withdraw()
     root.attributes('-topmost', True)
     folder = filedialog.askdirectory(title="📂 CV'lerin olduğu klasörü seçin")
     root.destroy()


     if not folder:
          print("⛔ Klasör seçilmedi. Program sonlandırılıyor.")
          exit()

     print(f"🔍 DEBUG: Seçilen klasör: {folder}")

     #Dosya sayısı kontrolü
     pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
     file_count = len(pdf_files)

     print(f"🔍 DEBUG: Bulunan PDF sayısı: {file_count}")
     print(f"🔍 DEBUG: PDF dosyaları: {pdf_files[:5]}...")

     if file_count == 0:
          print("⛔ Klasörde PDF dosyası bulunamadı.")
          exit()

     if file_count > 100:
          print(f"⚠️ {file_count} adet dosya bulundu. İşlem uzun sürebilir.")
          devam = input("Devam etmek istiyor musun? (e/h): ").lower()
          if devam != 'e':
               print("İşlem iptal edildi.")
               exit()


     #Eşik Değeri
     print("\n🎯 Benzerlik eşiği ayarları:")
     print("• 0.1 = Çok çok Esnek")
     print("• 0.2 = Çok Esnek")
     print("• 0.3 = Esnek")
     print("• 0.4 = Orta(Önerilen)")
     print("• 0.5 = Sıkı")
     print("• 0.6 = Çok Sıkı" )

     try:
          threshold = float(input("\n📊 Benzerlik eşiğini girin (0.1-0.6 arası önerilen): ").strip())
          if threshold < 0.05 or threshold > 0.9:
               print("⚠️ Geçersiz aralık. Eşik 0.4 olarak ayarlandı.")
               threshold = 0.4
     except:
          print("⚠️ Geçersiz değer, 0.4 kullanılıyor.")
          threshold = 0.4

     print(f"🔍 DEBUG: Kullanılacak threshold: {threshold}")


     try:
          prompt_embedding = model.encode(prompt, convert_to_tensor=True, show_progress_bar=False)
          print(f"✅ Prompt embedding başarıyla hesaplandı. Boyut: {prompt_embedding.shape}")
     except Exception as e:
          print(f"❌ Prompt embedding hatası: {e}")
          import traceback
          traceback.print_exc()
          exit()

     print(f"\n🚀 Analiz başlatılıyor... (Eşik: {threshold:.3f})")
     start_time = time()

     try:
          all_results, uygun_cv_listesi = analyze_cv_folder(prompt_embedding, threshold, folder)
          elapsed_time = time() - start_time

          print(f"\n{'='*60}")
          print(f"\n📊 İşlem Tamamlandı! ({elapsed_time:.1f})")
          print(f"\n{'='*60}")

          uygun_sayisi = len([r for r in all_results if r[2] == '✅ Uygun'])
          uygun_degil_sayisi = len([r for r in all_results if r[2] == '❌ Uygun Değil'])

          print(f"\nToplam CV: {len(all_results)}")
          print(f"Uygun: {uygun_sayisi}")
          print(f"Uygun Değil: {uygun_degil_sayisi}")      
          print(f"Kullanılan Eşik: {threshold:.3f}")

          print(f"\n🔍 DEBUG: Uygun CV listesi uzunluğu: {len(uygun_cv_listesi)}")
          if uygun_cv_listesi:
              print(f"🔍 DEBUG: Uygun CV'ler: {[cv[0] for cv in uygun_cv_listesi]}")

          print(f"\n📋 En İyi 10 Sonuç (Skor sırasına göre):")
          sorted_results = sorted(all_results, key = lambda x: x[1], reverse=True)
          for i, (fname, skor, durum) in enumerate(sorted_results[:10], 1):
               try:
                    #fname tuple değilse direkt yaz, tuple'sa düzelt
                    if isinstance(fname, tuple):
                         fname = fname[0]
                    print(f"{i:2d}. {fname[:50]:<50} -> Skor: {skor:.3f} {durum}")
               except Exception as e:
                    print(f"{i:2d}. <Dosya okunamadı> -> Skor: {skor:.3f} {durum} ({e})")

           # Tüm skorları göster
          print(f"\n🔍 DEBUG: Tüm skorlar:")
          for fname, skor, durum in sorted_results:
              print(f"  {fname}: {skor:.4f}")

          #Kopyalama Seçeneği
          if len(uygun_cv_listesi) > 0:
               copy_choice = input("\n📁 Uygun CV'leri ayrı klasöre kopyalamak ister misiniz? (e/h): ").lower()
               if copy_choice == 'e':
                    output_folder = os.path.join(folder, "Uygun_CVler")
                    os.makedirs(output_folder, exist_ok=True)

                    copied_count = 0
                    for fname, skor in uygun_cv_listesi:
                         try:
                              src = os.path.join(folder, fname)
                              dst = os.path.join(output_folder, fname)
                              copy2(src, dst)
                              copied_count += 1
                              print(f"📋 {fname} kopyalandı")
                         except Exception as e:
                              print(f"❌ Kopyalama hatası {fname}: {e}")

               print(f"\n✅ {copied_count} CV '{output_folder}' klasörüne kopyalandı!")
          else:
               print("\n📁 Kopyalama işlemi yapılmadı.")
             

     except Exception as e:
          print(f"\n❌ Analiz sırasında hata: {e}")
          print("💡 Lütfen dosya sayısını azaltın veya programı yeniden başlatın.")

     print(f"\n 🎯 Program tamamlandı!")

               











            
