import os
import torch
import lizard
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from src.extractors.complexity_analyzer import estimate_complexity

# Eğer model farklı bir klasördeyse import'u ayarlayın
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.architecture import CodeAnalyzerModel

app = FastAPI(title="Akıllı Kod Analizörü API", version="2.0")

class AnalyzeRequest(BaseModel):
    code: str
    ai_enabled: bool = True

class ModelSingleton:
    _model = None
    _tokenizer = None
    _device = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
            cls._model = CodeAnalyzerModel(
                pretrained_model_name="microsoft/graphcodebert-base",
                num_cwe_classes=50,
                use_memory=True,
                use_hierarchical_attention=True,
                gradient_checkpointing=False
            )
            model_path = os.path.join("checkpoints", "best_model.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=cls._device)
                if "model_state_dict" in checkpoint:
                    cls._model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    cls._model.load_state_dict(checkpoint)
            
            cls._model.to(cls._device)
            cls._model.eval()
        return cls._model, cls._tokenizer, cls._device


@app.post("/analyze")
async def analyze_code(request: AnalyzeRequest):
    code = request.code
    
    if not code or len(code.strip()) == 0:
        raise HTTPException(status_code=400, detail="Kod bos olamaz.")

    # 1. Statik Analiz (Lizard)
    try:
        analysis = lizard.analyze_file.analyze_source_code("source.cpp", code)
        funcs = analysis.function_list
        if funcs:
            max_ccn = max(f.cyclomatic_complexity for f in funcs)
            avg_ccn = sum(f.cyclomatic_complexity for f in funcs) / len(funcs)
            func_count = len(funcs)
        else:
            max_ccn = 0
            avg_ccn = 0.0
            func_count = 0
        nloc = analysis.nloc
        
        static_issues = []
        if max_ccn > 15:
            static_issues.append(f"Yuksek Cyclomatic Complexity tespit edildi: {max_ccn} (Onerilen max: 15)")
        if nloc > 500:
            static_issues.append(f"Dosya cok uzun: {nloc} satir. Parcalara bolunmesi onerilir.")

    except Exception as e:
        max_ccn = 0
        avg_ccn = 0.0
        func_count = 0
        nloc = 0
        static_issues = [f"Statik analiz hatasi: {e}"]

    response_data = {
        "static_analysis": {
            "total_files_analyzed": 1,
            "issues": static_issues,
            "metrics": {
                "max_cyclomatic_complexity": max_ccn,
                "average_cyclomatic_complexity": round(avg_ccn, 2),
                "function_count": func_count,
                "nloc": nloc
            }
        },
        "performance": {
            "status": "Iyi" if max_ccn < 10 else "Gelistirilebilir",
            "performance_warnings": 1 if max_ccn > 15 else 0,
            "metrics": [
                f"Fonksiyon Sayisi: {func_count}",
                f"Satir Sayisi (NLOC): {nloc}",
                f"Karmaşıklık: {max_ccn}"
            ]
        }
    }

    # 2. Yapay Zeka Analizi (GraphCodeBERT)
    if request.ai_enabled:
        try:
            model, tokenizer, device = ModelSingleton.get_model()
            
            inputs = tokenizer(
                code, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding="max_length"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    update_memory=False
                )
                
                vuln_logits = outputs["vulnerability_logits"]
                df_logits = outputs["dataflow_logits"]
                
                vuln_probs = torch.softmax(vuln_logits, dim=-1)
                vuln_class = torch.argmax(vuln_probs, dim=-1).item()
                is_vulnerable = vuln_class != 0
                
                # Zafiyet olasiligini skor olarak 10 uzerinden dondurelim (Orn: prob * 10)
                # Eger vuln_class != 0 ise, o sinifin olasiligi x 10, degilse (1 - no_vuln_prob) * 10
                if is_vulnerable:
                    severity = round(float(vuln_probs[0][vuln_class].item()) * 10, 1)
                else:
                    severity = round(float(1.0 - vuln_probs[0][0].item()) * 10, 1)

                df_probs = torch.softmax(df_logits, dim=-1)
                df_class = torch.argmax(df_probs, dim=-1).item()
                has_data_flow = df_class == 1

                observations = []
                if is_vulnerable:
                    observations.append(f"GraphCodeBERT Modeli CWE Zafiyeti tespit etti (Sinif Indexi: {vuln_class}).")
                if has_data_flow:
                    observations.append("Model guvensiz veri akisi (Taint) tespit etti.")
                else:
                    observations.append("Tehlikeli bir veri akisi saptanmadi.")

                response_data["ai_analysis"] = {
                    "is_defective_predicted": is_vulnerable,
                    "severity_score": severity,
                    "observations": observations
                }
        except Exception as e:
            response_data["ai_analysis"] = {
                "error": str(e)
            }

    return response_data
