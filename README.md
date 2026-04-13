# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

**Disciplina:** Inteligência Artificial / LLMs  
**Instituição:** iCEV — Instituto de Ensino Superior  
**Domínio:** Previsões Climáticas e Meteorologia  
**Modelo base:** `meta-llama/Llama-2-7b-hf`  
**Versão:** v1.0

---

## Objetivo

Pipeline completo de *fine-tuning* do **Llama 2 7B** com:
- **QLoRA** — quantização 4-bit (nf4 / float16) via `bitsandbytes`
- **LoRA** — adaptadores de baixo rank via `peft`
- **SFTTrainer** — treinamento supervisionado via `trl`
- **Dataset sintético** — gerado com a API OpenAI (GPT-3.5-turbo)

---

## Estrutura do Repositório

```
lab07-qlora/
├── step1_generate_dataset.py     # Passo 1: geração do dataset sintético
├── step2_3_4_finetune.py         # Passos 2-4: quantização, LoRA, treinamento
├── inference.py                  # Inferência com o adaptador salvo
├── lab07_qlora.ipynb             # Notebook completo para Google Colab ⭐
├── dataset_train.jsonl           # Dataset de treino (90%)
├── dataset_test.jsonl            # Dataset de teste (10%)
├── requirements.txt              # Dependências com versões fixas
└── README.md
```

---

## Configurações Implementadas

### Passo 2 — Quantização (QLoRA / BitsAndBytes)

| Parâmetro | Valor |
|---|---|
| `load_in_4bit` | `True` |
| `bnb_4bit_quant_type` | `"nf4"` (NormalFloat 4-bit) |
| `bnb_4bit_compute_dtype` | `torch.float16` |
| `bnb_4bit_use_double_quant` | `True` |

### Passo 3 — LoRA (PEFT)

| Parâmetro | Valor | Descrição |
|---|---|---|
| `r` (Rank) | **64** | Dimensão das matrizes menores |
| `lora_alpha` | **16** | Fator de escala dos novos pesos |
| `lora_dropout` | **0.1** | Evita overfitting |
| `task_type` | `CAUSAL_LM` | Modelagem de linguagem causal |

### Passo 4 — Treinamento (SFTTrainer)

| Parâmetro | Valor | Motivo |
|---|---|---|
| `optim` | `paged_adamw_32bit` | Transfere picos de memória GPU → CPU |
| `lr_scheduler_type` | `cosine` | LR decai suavemente em curva cosseno |
| `warmup_ratio` | `0.03` | Primeiros 3% do treino: LR sobe gradualmente |
| `learning_rate` | `2e-4` | — |
| `fp16` | `True` | Meia precisão para economizar VRAM |

---

## Como Executar

### Opção A — Google Colab (recomendado)

1. Abra `lab07_qlora.ipynb` no Google Colab
2. Vá em **Runtime → Change runtime type → GPU** (T4 ou A100)
3. Execute as células em ordem

### Opção B — Linha de Comando

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar variáveis de ambiente
export OPENAI_API_KEY="sua-chave-openai"
export HF_TOKEN="seu-token-huggingface"   # necessário para Llama 2 (modelo gated)

# 3. Gerar o dataset (≥ 50 pares)
python step1_generate_dataset.py

# 4. Executar o fine-tuning
python step2_3_4_finetune.py

# 5. Testar o modelo (opcional)
python inference.py
```

---

## Dataset

| Item | Detalhe |
|---|---|
| Domínio | Previsões Climáticas e Meteorologia |
| Formato | `.jsonl` com chaves `"prompt"` e `"response"` |
| Pares totais | ≥ 50 (60 gerados, filtrados por validação) |
| Divisão | 90% treino / 10% teste |
| Gerador | OpenAI GPT-3.5-turbo |

**Exemplos de tópicos cobertos:**
El Niño / La Niña, modelos GFS e ECMWF, índices CAPE e LI, frentes frias/quentes,
ciclones, tornados, radar meteorológico, ilhas de calor, mudanças climáticas, ZCIT e outros.

---

## Declaração de Uso de IA

> **Partes geradas/complementadas com IA, revisadas por [Seu Nome].**

Ferramentas de IA generativa foram utilizadas como apoio para:
- Pesquisa preliminar de boas práticas de QLoRA e hiperparâmetros
- Geração de templates de código submetidos à revisão crítica

Todo o código foi revisado, compreendido e adaptado para atender às especificações do laboratório.

---

## Referências

- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL — SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
