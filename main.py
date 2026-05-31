import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load
from jiwer import wer

# Load data
dataset = load_dataset("google/fleurs", "ka_ge")
print(dataset)

# Load model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
print("Loaded OK")

# Baseline evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

test_samples = dataset["test"].select(range(50))
references = []
hypotheses = []

for sample in test_samples:
    audio = sample["audio"]
    audio_array = audio["array"]
    sampling_rate = audio["sampling_rate"]
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    references.append(sample["transcription"])
    hypotheses.append(transcription)

baseline_wer = wer(references, hypotheses)
print(f"Baseline WER on Georgian: {baseline_wer:.4f} ({baseline_wer * 100:.2f}%)")

# Reload processor with language settings
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Georgian", task="transcribe")
model.generation_config.language = "georgian"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Preprocess dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])
print("Preprocessing done")

def filter_long_labels(example):
    return len(example["labels"]) <= 448

dataset = dataset.filter(filter_long_labels)
print("After filtering:", dataset)

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Metrics
metric = load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": metric.compute(predictions=pred_str, references=label_str)}

# Training
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ka",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=50,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

trainer.train()