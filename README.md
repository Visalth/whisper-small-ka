# Whisper Small - Georgian Fine-tune

Fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the Georgian language using the [Google FLEURS](https://huggingface.co/datasets/google/fleurs) dataset.

## Results

| Model | WER |
|-------|-----|
| Whisper Small (baseline) | 355.52% |
| Whisper Small Georgian (fine-tuned) | 57.14% |

**298 percentage point improvement** on the Georgian test set.

## Dataset

- **Training:** 1,281 samples from FLEURS `ka_ge`
- **Validation:** 352 samples
- **Test:** 768 samples

## Training

- **Base model:** openai/whisper-small
- **Hardware:** NVIDIA GeForce GTX 1070 (8GB VRAM)
- **Training steps:** 3,000 (best checkpoint)
- **Batch size:** 4 (effective batch size 16 with gradient accumulation)
- **Learning rate:** 1e-5 with linear scheduler
- **Training time:** ~10 hours

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Georgian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("irakli/whisper-small-ka")

inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
predicted_ids = model.generate(**inputs)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
```

## Limitations

- Small dataset (~10 hours of audio) limits generalization
- Best results on clean, studio-quality Georgian audio
- Trained on a single speaker variety — may not generalize well to all dialects

## Author

Irakli Kacitadze