from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from argparse import ArgumentParser

model = AutoModelForSequenceClassification.from_pretrained(
    "norsu/lean-ai-text-detector"
)
tokenizer = AutoTokenizer.from_pretrained("norsu/lean-ai-text-detector")


pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()
    print(pipe(args.input))
