import torch
from transformers import AutoModelForCausalLM, AutoProcessor


# NOTE: transformers==4.46.3 is recommended for this script
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


@torch.inference_mode()
def infer(conversation):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# Video conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": "/workspace/VideoLLaMA3/assets/cat_and_chicken.mp4", "fps": 24, "max_frames": 180}},
            {"type": "text", "text": "What is the cat doing? Please describe the scene, the obejcts and the actions in detail."},
        ]
    },
]
print(infer(conversation))

# Image conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": {"image_path": "/workspace/VideoLLaMA3/assets/sora.png"}},
            {"type": "text", "text": "Please describe the model?"},
        ]
    }
]
print(infer(conversation))

# Mixed conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": "/workspace/VideoLLaMA3/assets/cat_and_chicken.mp4", "fps": 1, "max_frames": 180}},
            {"type": "text", "text": "What is the relationship between the video and the following image?"},
            {"type": "image", "image": {"image_path": "/workspace/VideoLLaMA3/assets/sora.png"}},
        ]
    }
]
print(infer(conversation))

# Plain text conversation
conversation = [
    {
        "role": "user",
        "content": "What is the color of bananas?",
    }
]
print(infer(conversation))

from datasets import load_dataset
from datasets.features import Video

# Load normally (non-streaming)
dataset = load_dataset(
    "facebook/2M-Flores-ASL",
    split="dev",
    streaming=False
)

# Replace ONLY the video column decoder
dataset = dataset.cast_column("video", Video(decode=False))

# Now indexing will NOT trigger TorchCodec
example = dataset[0]

print(example.keys())         # all original columns preserved
print(example["video"])       # {'path': '...', 'bytes': None}
print(example["sentence"])

# transcribe ASL:

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": example["video"]["path"], "fps": 24, "max_frames": 180}},
            {"type": "text", "text": "What is being signed in the video?"},
        ]
    },
]
print(infer(conversation))
