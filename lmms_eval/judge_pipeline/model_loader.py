import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Auto-detect flash attention
try:
    import flash_attn
    ATTN_IMPL = "flash_attention_2"
    print(f"Using Flash Attention {flash_attn.__version__}")
except ImportError:
    ATTN_IMPL = "sdpa"
    print("Using PyTorch SDPA (flash-attn not available)")


def load_judge_model(model_id="Qwen/Qwen2.5-72B-Instruct"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
    )

    print(f"Loading model: {model_id} (4-bit NF4)")
    print(f"Attention: {ATTN_IMPL}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        torch_dtype=torch.bfloat16,
    )

    model.eval()

    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU Memory:")
    print(f"  Allocated: {allocated:.1f} GB")
    print(f"  Reserved:  {reserved:.1f} GB")
    print(f"  Total:     {total:.1f} GB")
    print(f"  Free:      {total - reserved:.1f} GB")

    return model, tokenizer
