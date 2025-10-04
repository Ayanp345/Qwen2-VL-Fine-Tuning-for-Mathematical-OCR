# Qwen2-VL-Fine-Tuning-for-Mathematical-OCR
This project demonstrates step-by-step fine-tuning of the open-source Qwen2 Vision-Language (VL) model to convert images of mathematical expressions into LaTeX code, using the Unsloth library for optimized training and speed.


Overview
Goal: Convert images of formulas into their corresponding LaTeX expressions.

Model: Qwen2-VL-7B-Instruct (vision-language, multimodal)

Dataset: LaTeX-OCR—pairs of images and LaTeX code.

Techniques: Parameter-efficient fine-tuning (LoRA), 4-bit quantization, customized data formatting for multimodal learning.

Workflow
1. Environment Setup
bash
# Set up environment
pip install "git+https://github.com/huggingface/transformers" accelerate peft bitsandbytes unsloth sentencepiece protobuf datasets
2. Model & Tokenizer Loading
python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)
3. Dataset Preparation
Use the LaTeX-OCR dataset.

For each sample, create a conversation-style dictionary:

Role: 'user', Content: includes image and prompt ("Write the LaTeX representation for this image")

Role: 'assistant', Content: the ground truth LaTeX code

python
from datasets import load_dataset

dataset = load_dataset('unsloth/LaTeX_OCR', split='train[:1000]')
def convert_to_conversation(sample):
    return {
        "conversations": [
            {"role": "user", "content": {"type": "image", "image": sample["image"], "text": "Write the LaTeX representation for this image."}},
            {"role": "assistant", "content": {"type": "text", "text": sample["text"]}}
        ]
    }

converted_dataset = dataset.map(convert_to_conversation)
4. Fine-Tuning with LoRA
python
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048
    )
)
trainer.train()
5. Inference and Validation
After fine-tuning, enable inference mode and test with images:

python
FastVisionModel.for_inference(model)
# Prepare a conversation as before, then...
from transformers import TextStreamer

messages = [ ... ] # As above, user/image + prompt and blank assistant
formatted_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(formatted_input, return_tensors='pt', add_special_tokens=False).to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True)
output_ids = model.generate(**inputs, max_new_tokens=120, streamer=streamer, do_sample=True)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)
Deployment
Platform: Google Colab, local GPU (T4/RTX series or better)

Requirements: PyTorch, Unsloth, Transformers (latest), datasets package, 8GB VRAM (minimum recommended)

Model Saving: After finetuning, save using .save_pretrained() and deploy using inference script above.

Tips: Convert conversation formatting during both training and inference for consistent results.

Key Takeaways
LoRA-based vision-language fine-tuning enables efficient domain adaptation with modest resources.

With 4-bit quantization, large VL models can run and train on mid-range GPUs.

Proper multimodal formatting (image-text pairs as conversations) is crucial for high OCR accuracy.

Links:

Original notebook

LaTeX OCR Dataset

Project video walkthrough
