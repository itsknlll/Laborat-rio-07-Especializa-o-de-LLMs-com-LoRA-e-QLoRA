"""
Inferência — testa o modelo fine-tunado no domínio climático.
Carrega o adaptador LoRA salvo e gera respostas meteorológicas.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL  = "meta-llama/Llama-2-7b-hf"
ADAPTER_DIR = "./llama2-clima-qlora-adapter"


def load_model_for_inference():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 300) -> str:
    prompt = (
        "### Instrução:\n"
        f"{instruction}\n\n"
        "### Resposta:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "### Resposta:" in decoded:
        return decoded.split("### Resposta:")[-1].strip()
    return decoded


if __name__ == "__main__":
    print("Carregando modelo fine-tunado (domínio climático)...")
    model, tokenizer = load_model_for_inference()

    test_prompts = [
        "O que é El Niño e como ele afeta as chuvas no Nordeste brasileiro?",
        "Explique o que é CAPE e como esse índice é usado na previsão de tempestades.",
        "Qual a diferença entre frente fria e frente quente?",
        "O que é inversão térmica e por que ela piora a qualidade do ar?",
        "Como funciona um radar meteorológico Doppler?",
    ]

    for prompt in test_prompts:
        print("\n" + "=" * 60)
        print(f"INSTRUÇÃO: {prompt}")
        print("-" * 60)
        print(f"RESPOSTA:\n{generate(model, tokenizer, prompt)}")
