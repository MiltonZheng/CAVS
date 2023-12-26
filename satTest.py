import argparse
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/visualglm-6b", trust_remote_code=True)
from model import chat, VisualGLMModel
model, model_args = VisualGLMModel.from_pretrained('visualglm-6b', args=argparse.Namespace(fp16=True, skip_init=True))
from sat.model.mixins import CachedAutoregressiveMixin
model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
image_path = "/root/autodl-tmp/visualglm-6b/cat.jpg"
response, history, cache_image = chat(image_path, model, tokenizer, "描述这张图片。", history=[])
print(response)
response, history, cache_image = chat(None, model, tokenizer, "这张图片可能是在什么场所拍摄的？", history=history, image=cache_image)
print(response)