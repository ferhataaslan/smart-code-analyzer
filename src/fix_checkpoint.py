import torch

checkpoint_path = "checkpoints/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

new_state_dict = {}
for key, value in state_dict.items():
    new_key = key
    
    # Isim degisikliklerini uygula
    if "hier_attn." in new_key:
        new_key = new_key.replace("hier_attn.", "hierarchical_attn.")
    if "level_pools." in new_key:
        new_key = new_key.replace("level_pools.", "level_attention.")
    if "cwe_head.network." in new_key:
        new_key = new_key.replace("cwe_head.network.", "vuln_head.classifier.")
    if "comp_head.network." in new_key:
        new_key = new_key.replace("comp_head.network.", "complexity_head.classifier.")
    if "df_head.network." in new_key:
        new_key = new_key.replace("df_head.network.", "dataflow_head.classifier.")
    if "vuln_head.network." in new_key:
        new_key = new_key.replace("vuln_head.network.", "vuln_head.classifier.")
    if "memory_bank._memory" in new_key:
        new_key = new_key.replace("memory_bank._memory", "memory_bank.memory")
    if "memory_bank._count" in new_key:
        new_key = new_key.replace("memory_bank._count", "memory_bank.memory_count")
    if "memory_bank.W_q." in new_key:
        new_key = new_key.replace("memory_bank.W_q.", "memory_bank.query_proj.")
    if "memory_bank.W_k." in new_key:
        new_key = new_key.replace("memory_bank.W_k.", "memory_bank.key_proj.")
    if "memory_bank.W_v." in new_key:
        new_key = new_key.replace("memory_bank.W_v.", "memory_bank.value_proj.")
    if "memory_bank.W_o." in new_key:
        new_key = new_key.replace("memory_bank.W_o.", "memory_bank.out_proj.")
        
    new_state_dict[new_key] = value

if "model_state_dict" in checkpoint:
    checkpoint["model_state_dict"] = new_state_dict
else:
    checkpoint = new_state_dict

torch.save(checkpoint, checkpoint_path)
print("Checkpoint basariyla guncellendi ve isim uyusmazliklari giderildi!")
