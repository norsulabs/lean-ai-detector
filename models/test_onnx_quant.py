import onnxruntime
import time
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

data = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("norsu/lean-ai-text-detector")


model_path = "lean-ai-text-detector-8bit.onnx"

sess_options = onnxruntime.SessionOptions()
session = onnxruntime.InferenceSession(model_path, sess_options)

st = time.time()

inputs = tokenizer(
    data["test"][:100]["text"], return_tensors="np", padding=True, truncation=True
)
ort_inputs = {
    "input_ids": inputs["input_ids"],
    "input_mask": inputs["attention_mask"],
}
outputs = np.argmax(np.reshape(session.run(None, ort_inputs), (-1, 2)), 1)
corr = 0
for i in range(len(outputs)):
    if outputs[i] == data["test"][i]["label"]:
        corr += 1
quant_acc = corr / len(outputs)
print(quant_acc)

quant_time = time.time() - st
print(quant_time)
