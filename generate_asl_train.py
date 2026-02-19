from datasets import load_dataset, Video
import json
from pathlib import Path

dataset = load_dataset("facebook/2M-Flores-ASL", split="dev", streaming=False)
dataset = dataset.cast_column("video", Video(decode=False))

out_path = Path("annotations_video.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for example in dataset:
        item = {
            "video": [example["video"]["path"]],
            "conversations": [
                {
                    "from": "human",
                    "value": "<video>\nPlease describe what is being signed in this video."
                },
                {
                    "from": "gpt",
                    "value": example["sentence"]  # ground-truth translation
                }
            ]
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
