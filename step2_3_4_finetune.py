"""
Laboratório 07 — Fine-Tuning com LoRA e QLoRA
Domínio: Previsões Climáticas e Meteorologia
Modelo base: meta-llama/Llama-2-7b-hf
Ambiente: Google Colab (T4/A100)

Passos implementados:
  Passo 2 - Configuração da Quantização (BitsAndBytesConfig, nf4, float16)
  Passo 3 - Arquitetura LoRA (r=64, alpha=16, dropout=0.1, CAUSAL_LM)
  Passo 4 - Pipeline de Treinamento (SFTTrainer, paged_adamw_32bit, cosine, warmup=0.03)
"""

import os
import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
TRAIN_FILE = "dataset_train.jsonl"
TEST_FILE = "dataset_test.jsonl"
OUTPUT_DIR = "./llama2-clima-qlora-adapter"
MAX_SEQ_LENGTH = 512
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # token HuggingFace para modelo gated


# ---------------------------------------------------------------------------
# Passo 2 — Configuração da Quantização (QLoRA)
# ---------------------------------------------------------------------------
def build_bnb_config() -> BitsAndBytesConfig:
    """
    Carrega o modelo base em 4-bits com NormalFloat (nf4).
    compute_dtype=float16 mantém operações numéricas em meia precisão.
    double_quant reduz ainda mais o uso de memória.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",                    # NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.float16,          # compute dtype: float16
        bnb_4bit_use_double_quant=True,
    )
    return bnb_config


# ---------------------------------------------------------------------------
# Passo 3 — Arquitetura LoRA
# ---------------------------------------------------------------------------
def build_lora_config() -> LoraConfig:
    """
    LoRA congela os pesos originais e treina apenas matrizes de decomposição.
    Hiperparâmetros obrigatórios conforme laboratório:
      r=64, lora_alpha=16, lora_dropout=0.1, task_type=CAUSAL_LM
    """
    lora_config = LoraConfig(
        r=64,                           # Rank: dimensão das matrizes menores
        lora_alpha=16,                  # Alpha: fator de escala dos novos pesos
        lora_dropout=0.1,               # Dropout: evita overfitting
        bias="none",
        task_type=TaskType.CAUSAL_LM,   # Tarefa: modelagem de linguagem causal
        target_modules=[                # módulos de atenção e MLP a adaptar
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return lora_config


# ---------------------------------------------------------------------------
# Passo 4 — Argumentos de Treinamento
# ---------------------------------------------------------------------------
def build_training_args() -> TrainingArguments:
    """
    Otimizador, scheduler e warmup obrigatórios pelo laboratório:
      optim          = paged_adamw_32bit  (picos GPU → CPU)
      lr_scheduler   = cosine             (decaimento suave)
      warmup_ratio   = 0.03              (3% de aquecimento)
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # ── Obrigatórios ──────────────────────────────────────────
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # ── Aprendizado ───────────────────────────────────────────
        learning_rate=2e-4,
        # ── Precisão (otimizado para Colab T4/A100) ───────────────
        fp16=True,
        bf16=False,
        # ── Logging e salvamento ──────────────────────────────────
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        evaluation_strategy="epoch",
        # ── Eficiência ────────────────────────────────────────────
        group_by_length=True,
        gradient_checkpointing=True,    # economiza VRAM no Colab
        report_to="none",
    )
    return training_args


# ---------------------------------------------------------------------------
# Utilitários de dataset
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_instruction(example: dict) -> dict:
    """
    Formata cada par (prompt, response) no template de instrução.
    O SFTTrainer consome a coluna 'text'.
    """
    text = (
        "### Instrução:\n"
        f"{example['prompt']}\n\n"
        "### Resposta:\n"
        f"{example['response']}"
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Laboratório 07 — Fine-Tuning QLoRA")
    print("Domínio: Previsões Climáticas e Meteorologia")
    print("Modelo : meta-llama/Llama-2-7b-hf")
    print("=" * 60)

    # ── 1. Datasets ───────────────────────────────────────────────
    print("\n[1/6] Carregando datasets...")
    train_data = load_jsonl(TRAIN_FILE)
    test_data = load_jsonl(TEST_FILE)
    print(f"      Treino: {len(train_data)} | Teste: {len(test_data)}")

    train_dataset = Dataset.from_list([format_instruction(x) for x in train_data])
    test_dataset = Dataset.from_list([format_instruction(x) for x in test_data])

    # ── 2. Tokenizer ──────────────────────────────────────────────
    print(f"\n[2/6] Carregando tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 3. Quantização BitsAndBytes (Passo 2) ─────────────────────
    print("\n[3/6] Configurando quantização 4-bit nf4 / float16...")
    bnb_config = build_bnb_config()

    # ── 4. Modelo base quantizado ─────────────────────────────────
    print("\n[4/6] Carregando modelo base em 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    # ── 5. LoRA (Passo 3) ─────────────────────────────────────────
    print("\n[5/6] Aplicando LoRA (r=64, alpha=16, dropout=0.1)...")
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 6. Treinamento (Passo 4) ──────────────────────────────────
    print("\n[6/6] Iniciando treinamento com SFTTrainer...")
    training_args = build_training_args()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    trainer.train()

    # ── 7. Salva o adaptador LoRA (obrigatório — Passo 4) ─────────
    print(f"\n✅ Treinamento concluído! Salvando adaptador em: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("   Adaptador salvo com sucesso.")


if __name__ == "__main__":
    main()
