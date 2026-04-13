"""
Passo 1 - Engenharia de Dados Sintéticos
Domínio: Previsões Climáticas e Meteorologia
Gera dataset de instruções usando a API da OpenAI,
salva em formato .jsonl e divide em treino/teste (90/10).
"""

import os
import json
import random
import time
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
DOMAIN = "previsões climáticas e meteorologia"
N_SAMPLES = 60          # gera 60 para garantir >= 50 válidos após filtragem
TRAIN_RATIO = 0.90
TRAIN_FILE = "dataset_train.jsonl"
TEST_FILE = "dataset_test.jsonl"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# ---------------------------------------------------------------------------
# Prompt de sistema
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""Você é um meteorologista especialista em {DOMAIN}.
Gere EXATAMENTE um par JSON com as chaves "prompt" e "response".
- "prompt": uma pergunta ou instrução técnica clara sobre {DOMAIN} (em português).
- "response": uma resposta detalhada, correta e didática (em português), com dados, exemplos e explicações científicas quando aplicável.
Retorne APENAS o JSON puro, sem texto adicional, sem blocos de markdown."""

# ---------------------------------------------------------------------------
# Tópicos do domínio climático
# ---------------------------------------------------------------------------
SEED_TOPICS = [
    "fenômenos El Niño e La Niña e seus impactos no Brasil",
    "modelos numéricos de previsão do tempo (GFS, ECMWF)",
    "diferença entre tempo e clima",
    "formação e classificação de nuvens",
    "ciclones extratropicais e tropicais",
    "chuvas convectivas versus estratiformes",
    "índices de instabilidade atmosférica (CAPE, LI)",
    "frentes frias e frentes quentes",
    "umidade relativa do ar e ponto de orvalho",
    "inversão térmica e poluição atmosférica",
    "monções e ventos alísios",
    "mudanças climáticas e aquecimento global",
    "ilhas de calor urbanas",
    "precipitação e ciclo hidrológico",
    "radiação solar e balanço de energia na Terra",
    "pressão atmosférica e isobares",
    "tornados: formação e escala Fujita",
    "granizo: formação e condições necessárias",
    "seca meteorológica versus seca hidrológica",
    "satélites meteorológicos e interpretação de imagens",
    "radar meteorológico e leitura de refletividade",
    "índice de conforto térmico e sensação térmica",
    "chuva ácida: causas e efeitos",
    "ondas de calor e ondas de frio",
    "ZCIT (Zona de Convergência Intertropical)",
]


def generate_pair(client: OpenAI, topic: str) -> dict | None:
    """Chama a API e retorna um par {prompt, response} ou None em falha."""
    user_msg = f"Crie um par instrução/resposta técnica detalhada sobre o tema climático: {topic}."
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.85,
            max_tokens=700,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        pair = json.loads(raw)
        assert "prompt" in pair and "response" in pair, "Chaves ausentes"
        assert len(pair["prompt"]) > 15, "Prompt muito curto"
        assert len(pair["response"]) > 30, "Response muito curta"
        return pair
    except Exception as exc:
        print(f"  [WARN] Falha ao gerar par para '{topic}': {exc}")
        return None


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"Gerando {N_SAMPLES} pares de instrução/resposta")
    print(f"Domínio: {DOMAIN}\n")

    pairs: list[dict] = []
    topics_cycle = (SEED_TOPICS * ((N_SAMPLES // len(SEED_TOPICS)) + 2))[:N_SAMPLES]
    random.shuffle(topics_cycle)

    for i, topic in enumerate(topics_cycle):
        print(f"  [{i+1:02d}/{N_SAMPLES}] {topic}")
        pair = generate_pair(client, topic)
        if pair:
            pairs.append(pair)
        time.sleep(0.6)

    if len(pairs) < 50:
        raise RuntimeError(
            f"Apenas {len(pairs)} pares válidos gerados. Mínimo exigido: 50."
        )

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    def save_jsonl(data: list[dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_pairs, TRAIN_FILE)
    save_jsonl(test_pairs, TEST_FILE)

    print(f"\n{'='*50}")
    print(f"✅ Dataset gerado com sucesso!")
    print(f"   Domínio : {DOMAIN}")
    print(f"   Treino  : {len(train_pairs)} exemplos → {TRAIN_FILE}")
    print(f"   Teste   : {len(test_pairs)} exemplos → {TEST_FILE}")
    print(f"   Total   : {len(pairs)} pares válidos")


if __name__ == "__main__":
    main()
