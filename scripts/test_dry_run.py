import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
import json
import sys
import os

sys.path.append('c:/Users/aslan/OneDrive/Masaüstü/Akıllı Kod analizatörü G.2/Kodlar/model')

try:
    from train_colab import SmartCodeAnalyzerModel, MaskedMultiTaskLoss, TrainingConfig, CodeAnalysisDataset, _collate_batch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
    print('1. Veri Seti Yükleniyor...')
    ds = load_dataset('smart-code-analyzer-team/cpp-vulnerability-dataset', split='train', streaming=True)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
    
    print('2. Veri Yükleyici (DataLoader) Testi...')
    dataset_iter = iter(ds)
    samples = [next(dataset_iter) for _ in range(4)]
    
    class DummyDataset(CodeAnalysisDataset):
        def __init__(self):
            self._tokenizer = tokenizer
            self._max_length = 512
            self._hf_dataset = samples
            
    dummy_ds = DummyDataset()
    batch = _collate_batch([dummy_ds._build_sample(dummy_ds._tokenizer.encode("int main() {}"), dummy_ds._extract_labels(s)) for s in samples])
    print(f"Batch Input IDs Shape: {batch['input_ids'].shape}")
    
    print('3. Model ve Loss Testi...')
    cfg = TrainingConfig()
    cfg.use_memory = False
    
    # Fast load model by using eager init instead of AutoModel pretrained weights for this quick test
    model = SmartCodeAnalyzerModel(cfg).to('cpu')
    model.eval()
    
    outputs = model(batch["input_ids"], batch["attention_mask"], update_memory=False)
    print(f"Vulnerability Logits Shape: {outputs['vulnerability_logits'].shape}")
    print(f"CWE Logits Shape: {outputs['cwe_logits'].shape}")
    print(f"Risk Pred Shape: {outputs['risk_pred'].shape}")
    
    criterion = MaskedMultiTaskLoss(cfg).to('cpu')
    losses = criterion(outputs, batch["labels"])
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print('TEST BAŞARILI: Model ileri beslemesi ve kayıp hesaplaması çökmeden çalıştı.')
    
except Exception as e:
    import traceback
    traceback.print_exc()
