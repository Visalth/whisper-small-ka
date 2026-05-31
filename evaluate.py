import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Georgian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-ka")
# Or load from HuggingFace: WhisperForConditionalGeneration.from_pretrained("Visalth/whisper-small-ka")
model = model.to("cuda")
dataset = load_dataset("google/fleurs", "ka_ge")
test_samples = dataset["test"].select(range(50))

references = []
hypotheses = []

for sample in test_samples:
    audio = sample["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    references.append(sample["transcription"])
    hypotheses.append(transcription)

final_wer = wer(references, hypotheses)
print(f"Baseline WER: 355.52%")
print(f"Final WER: {final_wer*100:.2f}%")
print(f"Improvement: {355.52 - final_wer*100:.2f}%")