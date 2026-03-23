"""
ASR Evaluation for AutoEIT

Computes CER, WER, and SER for each model's transcriptions against
the known stimulus sentences. Also builds a consensus transcript
using majority voting (ROVER-style) across models.
"""

import sys
import json
import unicodedata
import re
from pathlib import Path
from jiwer import wer, cer, process_words, process_characters

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPTIONS_DIR = BASE_DIR / "output" / "transcriptions"
OUTPUT_DIR = BASE_DIR / "output" / "evaluation"

# The 30 stimulus sentences (without syllable counts)
STIMULI = [
    "Quiero cortarme el pelo",
    "El libro está en la mesa",
    "El carro lo tiene Pedro",
    "El se ducha cada mañana",
    "¿Qué dice usted que va a hacer hoy?",
    "Dudo que sepa manejar muy bien",
    "Las calles de esta ciudad son muy anchas",
    "Puede que llueva mañana todo el día",
    "Las casas son muy bonitas pero caras",
    "Me gustan las películas que acaban bien",
    "El chico con el que yo salgo es español",
    "Después de cenar me fui a dormir tranquilo",
    "Quiero una casa en la que vivan mis animales",
    "A nosotros nos fascinan las fiestas grandiosas",
    "Ella sólo bebe cerveza y no come nada",
    "Me gustaría que el precio de las casas bajara",
    "Cruza a la derecha y después sigue todo recto",
    "Ella ha terminado de pintar su apartamento",
    "Me gustaría que empezara a hacer más calor pronto",
    "El niño al que se le murió el gato está triste",
    "Una amiga mía cuida a los niños de mi vecino",
    "El gato que era negro fue perseguido por el perro",
    "Antes de poder salir él tiene que limpiar su cuarto",
    "La cantidad de personas que fuman ha disminuido",
    "Después de llegar a casa del trabajo tomé la cena",
    "El ladrón al que atrapó la policía era famoso",
    "Le pedí a un amigo que me ayudara con la tarea",
    "El examen no fue tan difícil como me habían dicho",
    "¿Serías tan amable de darme el libro que está en la mesa?",
    "Hay mucha gente que no toma nada para el desayuno",
]

AUDIO_IDS = ["038010_EIT-2A", "038011_EIT-1A", "038012_EIT-2A", "038015_EIT-1A"]
MODELS = ["openai", "crisper_whisper", "assemblyai_disfluencies"]


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison.
    - Lowercase
    - Unicode NFC normalization
    - Remove punctuation (including ¿ ¡)
    - Preserve Spanish accents (é, ñ, ü)
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Remove punctuation but keep letters (including accented), numbers, spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Remove underscores (counted as \w)
    text = text.replace("_", "")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_transcriptions(audio_id: str, model: str) -> list:
    """Load transcriptions for one audio file from one model."""
    path = TRANSCRIPTIONS_DIR / model / f"{audio_id}.json"
    if not path.exists():
        print(f"  WARNING: Missing {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return [entry["text"] for entry in data]


def evaluate_model(audio_id: str, model: str) -> dict:
    """Compute CER, WER, SER for one model on one audio file."""
    transcriptions = load_transcriptions(audio_id, model)
    if not transcriptions:
        return {}

    num_sentences = min(len(transcriptions), len(STIMULI))

    # Normalize both stimulus and transcription
    references = [normalize_text(STIMULI[i]) for i in range(num_sentences)]
    hypotheses = [normalize_text(transcriptions[i]) for i in range(num_sentences)]

    # Replace empty strings with a placeholder to avoid jiwer errors
    for i in range(num_sentences):
        if not hypotheses[i]:
            hypotheses[i] = "[empty]"
        if not references[i]:
            references[i] = "[empty]"

    # Compute metrics
    total_wer = wer(references, hypotheses)
    total_cer = cer(references, hypotheses)

    # Sentence Error Rate: % of sentences with any error
    sentence_errors = sum(1 for r, h in zip(references, hypotheses) if r != h)
    total_ser = sentence_errors / num_sentences

    # Per-sentence breakdown
    per_sentence = []
    for i in range(num_sentences):
        s_wer = wer(references[i], hypotheses[i])
        s_cer = cer(references[i], hypotheses[i])
        exact_match = references[i] == hypotheses[i]
        per_sentence.append({
            "sentence_num": i + 1,
            "stimulus": STIMULI[i],
            "transcription": transcriptions[i],
            "normalized_ref": references[i],
            "normalized_hyp": hypotheses[i],
            "wer": round(s_wer, 3),
            "cer": round(s_cer, 3),
            "exact_match": exact_match,
        })

    return {
        "audio_id": audio_id,
        "model": model,
        "num_sentences": num_sentences,
        "wer": round(total_wer, 3),
        "cer": round(total_cer, 3),
        "ser": round(total_ser, 3),
        "exact_matches": num_sentences - sentence_errors,
        "per_sentence": per_sentence,
    }


def build_consensus(audio_id: str) -> list:
    """Build a consensus transcript using majority voting across models."""
    all_transcriptions = {}
    for model in MODELS:
        transcriptions = load_transcriptions(audio_id, model)
        if transcriptions:
            all_transcriptions[model] = transcriptions

    if not all_transcriptions:
        return []

    num_sentences = min(len(t) for t in all_transcriptions.values())
    consensus = []

    for i in range(num_sentences):
        texts = {model: normalize_text(all_transcriptions[model][i]) for model in all_transcriptions}
        raw_texts = {model: all_transcriptions[model][i] for model in all_transcriptions}

        # Count agreement
        values = list(texts.values())
        unique = set(values)

        if len(unique) == 1:
            # Full agreement
            agreement = "full"
            best = list(raw_texts.values())[0]
        else:
            # Check for 2-of-3 agreement
            from collections import Counter
            counts = Counter(values)
            most_common, count = counts.most_common(1)[0]
            if count >= 2:
                agreement = "partial"
                # Pick the raw text from the first model that matches
                for model in MODELS:
                    if texts[model] == most_common:
                        best = raw_texts[model]
                        break
            else:
                agreement = "none"
                # All disagree — pick OpenAI (best overall)
                best = raw_texts.get("openai", list(raw_texts.values())[0])

        consensus.append({
            "sentence_num": i + 1,
            "text": best,
            "agreement": agreement,
            "model_outputs": {model: raw_texts[model] for model in all_transcriptions},
        })

    return consensus


def run_evaluation():
    """Run full evaluation across all files and models."""
    print(f"{'=' * 70}")
    print("AutoEIT ASR Evaluation")
    print(f"{'=' * 70}")

    all_results = []
    summary = []

    for audio_id in AUDIO_IDS:
        print(f"\n--- {audio_id} ---")
        for model in MODELS:
            result = evaluate_model(audio_id, model)
            if result:
                all_results.append(result)
                print(f"  {model:<25} WER: {result['wer']:.3f}  CER: {result['cer']:.3f}  "
                      f"SER: {result['ser']:.3f}  Exact: {result['exact_matches']}/30")
                summary.append({
                    "audio_id": audio_id,
                    "model": model,
                    "wer": result["wer"],
                    "cer": result["cer"],
                    "ser": result["ser"],
                    "exact_matches": result["exact_matches"],
                })

    # Overall averages per model
    print(f"\n{'=' * 70}")
    print("Overall Averages (across all 4 files)")
    print(f"{'=' * 70}")
    for model in MODELS:
        model_results = [r for r in summary if r["model"] == model]
        if model_results:
            avg_wer = sum(r["wer"] for r in model_results) / len(model_results)
            avg_cer = sum(r["cer"] for r in model_results) / len(model_results)
            avg_ser = sum(r["ser"] for r in model_results) / len(model_results)
            total_exact = sum(r["exact_matches"] for r in model_results)
            print(f"  {model:<25} WER: {avg_wer:.3f}  CER: {avg_cer:.3f}  "
                  f"SER: {avg_ser:.3f}  Exact: {total_exact}/120")

    # Build consensus for each file
    print(f"\n{'=' * 70}")
    print("Consensus Analysis")
    print(f"{'=' * 70}")
    all_consensus = {}
    for audio_id in AUDIO_IDS:
        consensus = build_consensus(audio_id)
        all_consensus[audio_id] = consensus
        full = sum(1 for c in consensus if c["agreement"] == "full")
        partial = sum(1 for c in consensus if c["agreement"] == "partial")
        none_ = sum(1 for c in consensus if c["agreement"] == "none")
        print(f"  {audio_id}: Full agreement: {full}, Partial (2/3): {partial}, No agreement: {none_}")

    # Save everything
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Detailed results saved to {results_path}")

    # Save summary
    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to {summary_path}")

    # Save consensus
    consensus_path = OUTPUT_DIR / "consensus_transcripts.json"
    with open(consensus_path, "w", encoding="utf-8") as f:
        json.dump(all_consensus, f, indent=2, ensure_ascii=False)
    print(f"  Consensus saved to {consensus_path}")

    return all_results, summary, all_consensus


if __name__ == "__main__":
    run_evaluation()
