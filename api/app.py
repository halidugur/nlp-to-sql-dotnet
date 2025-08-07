import subprocess
import threading
import webbrowser
import time
import json
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import difflib

# === 1. Web Arayüzünü Başlat ===
def run_dotnet():
    try:
        subprocess.Popen(
            ["dotnet", "run"],
            cwd=r"C:\xxxxxx",
            shell=True
        )
    except Exception as e:
        print(f".NET arayüz başlatılamadı: {e}")

def open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:5296")

threading.Thread(target=run_dotnet, daemon=True).start()
threading.Thread(target=open_browser, daemon=True).start()

# === 2. Yol Ayarları ===
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_NLP = BASE_DIR / "src" / "nlp"
SRC_SQL = BASE_DIR / "src" / "query_builder"

sys.path.append(str(SRC_NLP))
sys.path.append(str(SRC_SQL))

# === 3. Modül Yükle ===
from nlp_processor import NLPProcessor
from sql_generator import SQLGenerator

# === 4. API Yapısı ===
app = FastAPI(
    title="Turkish NLP-SQL API",
    description="Doğal dilden SQL sorgusu üreten sistem",
    version="1.0"
)

class QueryRequest(BaseModel):
    text: str

nlp_processor = NLPProcessor()
sql_generator = SQLGenerator()

# === 5. Veri Setini Yükle (örnek eşleşme için) ===
data_path = BASE_DIR / "data" / "nl2sql_dataset_200k_tr.json"
if not data_path.exists():
    raise FileNotFoundError(f"Veri seti bulunamadı: {data_path}")

with open(data_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)
    known_samples = {entry["text"]: entry["sql"] for entry in json_data}

# === 6. Sorgu İşleme ===
@app.post("/generate-sql")
def generate_sql(req: QueryRequest):
    try:
        start = time.time()
        nlp_result = nlp_processor.analyze(req.text)
        sql_result = sql_generator.generate_sql(nlp_result)
        elapsed = round(time.time() - start, 3)

        if sql_result.get("success"):
            return {
                "success": True,
                "sql": sql_result["sql"],
                "intent": sql_result.get("intent"),
                "table": sql_result.get("table"),
                "confidence": sql_result.get("confidence"),
                "has_time_filter": sql_result.get("has_time_filter"),
                "elapsed": elapsed
            }
        else:
            best_match = difflib.get_close_matches(req.text, known_samples.keys(), n=1)
            if best_match:
                return {
                    "success": False,
                    "error": "Model SQL üretemedi.",
                    "did_you_mean": best_match[0],
                    "suggested_sql": known_samples[best_match[0]],
                    "elapsed": elapsed
                }
            return {
                "success": False,
                "error": "Model SQL üretemedi ve benzer sorgu bulunamadı.",
                "elapsed": elapsed
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")

# === 7. Ana Sayfa ===
@app.get("/")
def root():
    return {"message": "Turkish NLP-SQL API çalışıyor. POST /generate-sql ile sorgula."}
