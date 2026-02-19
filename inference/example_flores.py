import sys
sys.path.append('./')
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.mm_utils import load_video, load_images


def main():
    disable_torch_init()

    # from datasets import load_dataset, Video

    # # Load the dataset but override features to avoid Video
    # dataset = load_dataset(
    #     "facebook/2M-Flores-ASL",
    #     split="dev",
    #     streaming=False
    # )

    # print("bruh")
    
    # # Replace ONLY the video column decoder
    # # dataset = dataset.cast_column("video", Video(decode=False))
    
    # # Now indexing will NOT trigger TorchCodec
    # example = dataset[0]
    
    # print(example.keys())         # all original columns preserved
    # print(example["video"])       # {'path': '...', 'bytes': None}
    # print(example["sentence"])
    
    # transcribe ASL:

    modal = "video"
    frames, timestamps = load_video("/workspace/hf_home/hub/datasets--facebook--2M-Flores-ASL/snapshots/b450c1a427738e78f06362fc4619674f5d74f774/data/dev/dev_3_0.mov", fps=1, max_frames=180)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                {"type": "text", "text": "<video> Please describe what is being signed in this video."},
            ]
        }
    ]

    model_path = "./work_dirs/videollama3_qwen2.5_2b/stage_1"
    model, processor = model_init(model_path)
    model.to("cuda")

    inputs = processor(
        images=[frames] if modal != "text" else None,
        text=conversation,
        merge_size=2 if modal == "video" else 1,
        return_tensors="pt",
    )

    output = mm_infer(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal=modal
    )
    print(output)


if __name__ == "__main__":
    main()