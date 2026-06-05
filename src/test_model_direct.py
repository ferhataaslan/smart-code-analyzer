import os
import sys
import torch
from transformers import AutoTokenizer

# Proje dizinine erisim
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model.architecture import CodeAnalyzerModel

def load_model():
    print("Model yukleniyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = CodeAnalyzerModel(
        pretrained_model_name="microsoft/graphcodebert-base",
        num_cwe_classes=50,
        use_memory=True,
        use_hierarchical_attention=True,
        gradient_checkpointing=False
    )
    
    model_path = os.path.join("checkpoints", "best_model.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model basariyla yuklendi!")
    else:
        print(f"DIKKAT: {model_path} bulunamadi! Model rastgele agirliklarla calisacak.")
        
    model.to(device)
    model.eval()
    return model, tokenizer, device

def analyze(code, model, tokenizer, device):
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
        if is_vulnerable:
            severity = round(float(vuln_probs[0][vuln_class].item()) * 10, 2)
        else:
            severity = round(float(1.0 - vuln_probs[0][0].item()) * 10, 2)

        df_probs = torch.softmax(df_logits, dim=-1)
        df_class = torch.argmax(df_probs, dim=-1).item()
        has_data_flow = df_class == 1

        return {
            "vuln_class": vuln_class,
            "vuln_prob": float(vuln_probs[0][vuln_class].item()),
            "is_vulnerable": is_vulnerable,
            "severity_score": severity,
            "dataflow_class": df_class
        }

snippets = {
    "1. Kullanici Test Kodu (Deadlock)": """
#include <pthread.h>
#include <stdio.h>

volatile int a = 5;
volatile int b = 10;
pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;

void* worker_thread_bad(void* dummy) {
    int i;
    int result;
    if ((result = pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &i))!= 0) {
        return NULL;
    }
    while (1) {
        pthread_mutex_lock(&global_lock);
        int temp = a;
        a = b;
        b = temp;
        pthread_mutex_unlock(&global_lock);
    }
    return NULL;
}
""",
    "2. Buffer Overflow (CWE-121)": """
#include <string.h>
void bad_copy() {
    char dest[10];
    char src[] = "This is a very long string that will overflow";
    strcpy(dest, src);
}
""",
    "3. Format String Vulnerability (CWE-134)": """
#include <stdio.h>
void log_message(char *msg) {
    // Missing format specifier %s
    printf(msg); 
}
""",
    "4. Null Pointer Dereference (CWE-476)": """
#include <stdio.h>
#include <stdlib.h>
void do_something() {
    int *ptr = NULL;
    *ptr = 10; // Dereferencing NULL
}
""",
    "5. Safe Code (No Vuln)": """
#include <stdio.h>
int main() {
    printf("Hello, World!\\n");
    return 0;
}
""",
    "6. Integer Overflow (CWE-190)": """
#include <limits.h>
int add_numbers(int a, int b) {
    return a + b; // Potential overflow
}
void test() {
    add_numbers(INT_MAX, 1);
}
"""
}

def main():
    model, tokenizer, device = load_model()
    
    print("\n" + "="*50)
    print("MODEL TEST SONUCLARI")
    print("="*50)
    
    for name, code in snippets.items():
        print(f"\nTest Ediliyor: {name}")
        res = analyze(code, model, tokenizer, device)
        print(f" -> Sinif Indexi (CWE Class): {res['vuln_class']}")
        print(f" -> Zafiyet Var Mi?        : {res['is_vulnerable']}")
        print(f" -> Model Olasiligi        : {res['vuln_prob']:.4f}")
        print(f" -> UI Ciddiyet Skoru      : {res['severity_score']}")
        print(f" -> DataFlow Zafiyeti      : {res['dataflow_class'] == 1}")

if __name__ == "__main__":
    main()
