# import os

# AVAILABLE_MODELS = {
#     "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
#     "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
#     "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
#     "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
#     # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",
#     # Add other models as needed
# }
# print("Falling back to explicit import of LlavaLlamaForCausalLM")


# try:
#     from .llava_llama import LlavaLlamaForCausalLM
# except ImportError:
#     print("Failed to import LlavaLlamaForCausalLM from llava_llama module. This is import error ")
#     pass
# print("======================Falling back to explicit import of LlavaLlamaForCausalLM")


# # Try to import from language_model (official layout)
# for model_name, model_classes in AVAILABLE_MODELS.items():
#     try:
#         exec(f"from .language_model.{model_name} import {model_classes}")
#     except Exception as e:
#         # Fallback to local files (your repo layout)
#         try:
#             exec(f"from .{model_name} import {model_classes}")
#         except Exception as e2:
#             print(
#                 f"Failed to import {model_name} from llava.language_model.{model_name} "
#                 f"or llava.model.{model_name}. Error: {e2}"
#             )

# # === Explicit fallback imports ===
# # This ensures lmms_eval can always find LlavaLlamaForCausalLM directly.


import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",
}

# Try to import from the correct path (language_model submodule)
for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.model.language_model.{model_name}. Error: {e}")

# Explicitly expose LlavaLlamaForCausalLM for lmms_eval
try:
    from .language_model.llava_llama import LlavaLlamaForCausalLM
except ImportError as e:
    print(f"Explicit import of LlavaLlamaForCausalLM failed: {e}")
    LlavaLlamaForCausalLM = None
