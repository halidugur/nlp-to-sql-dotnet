NLP-to-SQL-dotnet Projesi Kurulum ve Kullanım Kılavuzu
1.  Gerekli Başlangıç Kurulumları
1.0 - Proje Dosyasını Hazırlayın
ZIP dosyasını çıkarın.
Klasörü açın.
1.1 - VS Code veya IDE ile Açın
code .

1.2 - Sanal Ortam Oluşturun (Ana dizinde)
Windows için:
python -m venv venv
venv\Scripts\activate

Linux/macOS:
python3 -m venv venv
source venv/bin/activate

1.3 - Gerekli Python Kütüphanelerini Kurun
Ana dizinde şu komutu çalıştırın:
pip install -r requirements.txt

2. Model Kurulumu ve Veri Seti Oluşturma
2.0 - Hugging Face Token Oluşturma ve Tanıtma
HuggingFace hesabına giriş yap: https://huggingface.co/settings/tokens
Yeni bir access token oluştur.

Terminale gir:
huggingface-cli login
İstenince token'ı yapıştırın.
2.1 - Modeli İndir
cd scripts
python download_models.py


3. Eğitim Verisi ve Model Eğitimi
3.0 - Eğitim Verisi Oluşturma (Opsiyonel)
Genellikle generate_ner_data.py dosyasına gerek yoktur çünkü eğitim verileri hazır gelir.
cd scripts
python generate_ner_data.py
3.1 - Veri Hazırlığı (data_processor.py)
cd ../src/nlp/ner_model
python data_processor.py
Bu dosya, eğitim için gerekli features, labels, ve dataset dosyalarını oluşturur.
3.2 - Model Eğitimi Başlatma (ner_trainer.py)
python ner_trainer.py
Eğitim süresi donanımınıza göre değişebilir.
<img width="1089" height="664" alt="trainer" src="https://github.com/user-attachments/assets/c99bc6ca-821a-4307-a772-049e347b30df" />
<img width="412" height="276" alt="image" src="https://github.com/user-attachments/assets/fdedbb13-ec5d-4434-b4eb-4fafd8091f2d" />

3.3 - NLP Processor Çalıştırma
cd ..
python nlp_processor.py
Eğitilmiş modeli yükleyerek metin girdilerini işler.

5. Deneme ve API Bağlantısı
4.0 - Komut Satırından NLP-to-SQL Test Et
Ana dizine dön:
cd ../../../
python turkish_nlp_sql_cli.py
Size doğal dilde soru sorabilir ve üretilen SQL sorgusunu görebilirsiniz.
<img width="1504" height="707" alt="Ekran Görüntüsü (451)" src="https://github.com/user-attachments/assets/2778220d-1d8a-4197-8e55-559ab3e68cc8" />
4.1 - Python'dan .NET Web API'yi Başlat
api/app.py dosyasındaki run_dotnet() fonksiyonunu şu şekilde güncelleyin:

def run_dotnet():
    try:
        subprocess.Popen(
            ["dotnet", "run"],
            cwd=r"C:\staj\nlp_sql_engine\WebApi",
            shell=True
        )
    except Exception as e:
        print(f".NET arayüz başlatılamadı: {e}")
4.2 - API'yi Başlat
Terminal üzerinden API’yi çalıştırmak için:

cd api
uvicorn app:app --reload
app.py içinde .NET servisini başlatır ve NLP modeline istek göndermek için API sunucusunu açar.

5. .NET Core (WebApi) ve Veritabanı Entegrasyonu
5.0 - WebApi/appsettings.json Dosyası
json
"ConnectionStrings": {
  "DefaultConnection": "Server=localhost;Database=nlp_db;Trusted_Connection=True;"
}
Buradaki bağlantı bilgisi, sizin SQL Server ayarlarınıza göre düzenlenmelidir. İsterseniz uzak bir SQL Server da kullanabilirsiniz.

Genel Akış Özeti
📁 Ana Dizin
│
├── venv                   ← Sanal ortam
├── requirements.txt       ← Python bağımlılıkları
├── turkish_nlp_sql_cli.py ← CLI arayüzü
│
├── api/
│   └── app.py             ← Uvicorn ile çalışan FastAPI sunucusu
│
├── WebApi/
│   └── appsettings.json   ← .NET Core ayarları
│
├── scripts/
│   ├── download_models.py
│   └── generate_ner_data.py (isteğe bağlı)
│
└── src/
    └── nlp/
        ├── nlp_processor.py
        └── ner_model/
            ├── data_processor.py
            └── ner_trainer.py
PROJE HAKKINDA DETAYLI BİLGİ İÇİN: https://www.notion.so/Text-to-SQL-Ara-t-rma-Ve-Proje-Raporu-2486487db46f8132ac55e19ce2f9c422
