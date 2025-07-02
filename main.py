import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import re
import unicodedata
import evaluate

# ✅ Model and Tokenizer
model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
print("✅ Model loaded on:", model.device)

# ✅ Download required NLTK resources
nltk.download('punkt_tab')

# ✅ Language name → ISO code (used by NLLB model)
LANG_CODE_MAP = {
    "english": "eng_Latn",
    "tamil": "tam_Taml",
    "hindi": "hin_Deva",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "telugu": "tel_Telu",
    "malayalam": "mal_Mlym",
    "arabic": "arb_Arab",
    "bengali": "ben_Beng",
    "chinese": "zho_Hans",
    "spanish": "spa_Latn"
}

# ✅ Preprocessing
def preprocess_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[?!]", ".", text)
    return text

# ✂️ Sentence segmentation
def segment_sentences(text):
    return nltk.sent_tokenize(text)

# 🧽 Postprocessing
def postprocess_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text

# 🌐 Translation using NLLB
def translate(text, src_lang_name, tgt_lang_name):
    src_code = LANG_CODE_MAP.get(src_lang_name.lower())
    tgt_code = LANG_CODE_MAP.get(tgt_lang_name.lower())

    if not src_code or not tgt_code:
        raise ValueError("❌ Unsupported language name.")

    tokenizer.src_lang = src_code
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    output = model.generate(
        **inputs,
        forced_bos_token_id=tgt_lang_id,
        max_length=512
    )

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

# 🔄 Clean and Translate Full Paragraph
def clean_and_translate(paragraph, src_lang, tgt_lang):
    preprocessed = preprocess_text(paragraph)
    sentences = segment_sentences(preprocessed)

    translated_sentences = []
    for sent in sentences:
        translated = translate(sent, src_lang, tgt_lang)
        postprocessed = postprocess_text(translated)
        translated_sentences.append(postprocessed)

    return " ".join(translated_sentences)

# 📊 Evaluation
evaluate_bleu = evaluate.load("bleu")
evaluate_chrf = evaluate.load("chrf")

def evaluate_translation(prediction, reference):
    print("\n📊 Evaluation Metrics:")

    bleu_result = evaluate_bleu.compute(predictions=[prediction], references=[[reference]])
    print(f"🔵 BLEU Score: {bleu_result['bleu']*100:.2f}")

    chrf_result = evaluate_chrf.compute(predictions=[prediction], references=[reference])
    print(f"🟢 CHRF Score: {chrf_result['score']:.2f}")

# 🌍 Interactive Translation Interface
print("\n🌍 Multilingual Translator using NLLB-200 1.3B")
src_lang = input("Enter source language (e.g., English): ").strip()
tgt_lang = input("Enter target language (e.g., Tamil): ").strip()

print("\n📥 Paste your paragraph to translate (press Enter twice to finish):")
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)
paragraph = "\n".join(lines)

translated_output = clean_and_translate(paragraph, src_lang, tgt_lang)

print("\n✅ Translated Output:\n")
print(translated_output)

# 🔁 Optional: Evaluate with Human Reference
# reference_text = input("\n📝 Paste the correct translation (reference):\n")
# evaluate_translation(translated_output, reference_text)
