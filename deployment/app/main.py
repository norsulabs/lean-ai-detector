from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints
from typing import List
from typing_extensions import Annotated
from transformers import AutoTokenizer
import onnxruntime
from huggingface_hub import hf_hub_download
from scipy.special import softmax

import re


def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


tokenizer = AutoTokenizer.from_pretrained(
    "norsu/lean-ai-text-detector",
)


sess_options = onnxruntime.SessionOptions()


model = hf_hub_download(
    repo_id="norsu/lean-ai-text-detector",
    filename="lean-ai-text-detector-8bit.onnx",
)
session = onnxruntime.InferenceSession(
    model,
    sess_options,
)


app = FastAPI()


class TextRequest(BaseModel):
    text: Annotated[
        str,
        StringConstraints(min_length=1, strip_whitespace=True),
    ]


@app.post("/predict/")
async def predict_sentiment(request: TextRequest):
    try:
        inputs = tokenizer(
            clean_text(request.text),
            padding=True,
            truncation=True,
        )
        ort_inputs = {
            "input_ids": [inputs["input_ids"]],
            "input_mask": [inputs["attention_mask"]],
        }
        result = session.run(None, ort_inputs)

        predicted_label = softmax(result[0])[0][0]

        return {"result": predicted_label.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
