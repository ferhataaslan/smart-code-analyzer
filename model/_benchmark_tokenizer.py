import time
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

# Sahte bir kod oluşturalım, yaklaşık 1000 token
code = "int main() { \n" + "printf('hello');\n" * 200 + "return 0;\n}"

start = time.time()
n = 62369

print(f"Tokenizing {n} records...")
for i in range(100):  # Sadece 100 tanesini deneyelim
    tokenizer(code, truncation=False, add_special_tokens=True, return_attention_mask=False)

end = time.time()
print(f"100 records took {end-start:.2f} seconds")
print(f"Estimated for {n}: {(end-start) * n / 100:.2f} seconds")
