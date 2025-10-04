# Qwen2-VL-Fine-Tuning-for-Mathematical-OCR
This project demonstrates step-by-step fine-tuning of the open-source Qwen2 Vision-Language (VL) model to convert images of mathematical expressions into LaTeX code, using the Unsloth library for optimized training and speed.


Overview
Goal: Convert images of formulas into their corresponding LaTeX expressions.

Model: Qwen2-VL-7B-Instruct (vision-language, multimodal)

Dataset: LaTeX-OCR—pairs (https://huggingface.co/datasets/unsloth/LaTeX_OCR) of images and LaTeX code.

Techniques: Parameter-efficient fine-tuning (LoRA), 4-bit quantization, customized data formatting for multimodal learning.
