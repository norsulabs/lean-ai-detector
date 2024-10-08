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
    # Remove all symbols except for alphanumeric characters and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing spaces
    text = text.strip()

    return text




# Initialize the sentiment analysis pipeline
tokenizer = AutoTokenizer.from_pretrained(
    "norsu/lean-ai-text-detector",
)


# model_path = r"lean-ai-text-detector-8bit.onnx"

sess_options = onnxruntime.SessionOptions()


model = hf_hub_download(
    repo_id="norsu/lean-ai-text-detector",
    filename="lean-ai-text-detector-8bit.onnx",
)
session = onnxruntime.InferenceSession(
    model,
    sess_options,
)


# Initialize FastAPI app
app = FastAPI()


# Define a request model to ensure valid input with Annotated
class TextRequest(BaseModel):
    text: Annotated[
        str,
        StringConstraints(min_length=1, strip_whitespace=True),
    ]


# Define a batch request model to handle multiple texts at once with Annotated


@app.post("/predict/")
async def predict_sentiment(request: TextRequest):
    try:
        inputs = tokenizer(
            clean_text(request.text),
            padding=True,
            truncation=True,
        )
        print(inputs)
        ort_inputs = {
            "input_ids": [inputs["input_ids"]],
            "input_mask": [inputs["attention_mask"]],
        }
        print(ort_inputs)
        # Run inference on the model
        result = session.run(None, ort_inputs)

        # Take the argmax to get the predicted label
        predicted_label = softmax(result[0])[0][0]

        return {"result": predicted_label.tolist()}
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
