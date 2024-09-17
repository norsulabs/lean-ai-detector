from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model
from huggingface_hub import HfApi

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased"
)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
peft_config = "./output/checkpoint-2625/"


model_to_merge = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased"
    ),
    peft_config,
)

merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")


api = HfApi()

api.upload_folder(
    folder_path="merged_model",  # Path to your local merged model
    repo_id="norsu/lean-ai-text-detector",
    repo_type="model",
)
