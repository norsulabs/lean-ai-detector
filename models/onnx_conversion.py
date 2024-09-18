import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "norsu/lean-ai-text-detector"
)
tokenizer = AutoTokenizer.from_pretrained("norsu/lean-ai-text-detector")

batch_size = 32

with torch.inference_mode():
    inputs = {
        "input_ids": torch.ones(1, 512, dtype=torch.int64),
        "attention_mask": torch.ones(1, 512, dtype=torch.int64),
    }
    outputs = model(**inputs)
    symbolic_names = {0: "batch_size", 1: "max_seq_len"}
    torch.onnx.export(
        model,
        (
            inputs["input_ids"],
            inputs["attention_mask"],
        ),
        r"lean-ai-text-detector.onnx",
        opset_version=12,
        input_names=[
            "input_ids",
            "input_mask",
        ],
        output_names=["output"],
        dynamic_axes={
            "input_ids": symbolic_names,
            "input_mask": symbolic_names,
        },
    )
