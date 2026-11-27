#!/usr/bin/env python3
"""Tokenizer goldens from HF `tokenizers` with the model's own tokenizer.json (model-free oracle)
-> tests/golden/tokenize_<name>.json, same shape as tokenize.json (parse_special true: HF always
splits added tokens).

Usage: uv run --with tokenizers python tools/capture_tokenize_hf.py openai/gpt-oss-20b gpt-oss
       (also zai-org/GLM-4.7-Flash glm, Qwen/Qwen3.6-35B-A3B qwen36)
"""
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

repo, name = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_pretrained(repo)
root = Path(__file__).resolve().parent.parent / "tests" / "golden"
texts = [c["text"] for c in json.load(open(root / "tokenize.json"))]
texts += [
    "DON'T you KNOW it's 1234567 o'clock? They'll've we'D", "HTTPServerError camelCaseIdentifier XMLHttpRequest",
    "日本語のテキスト、中文文本。한국어", "emoji 🚀🔥 and café naïve façade", "tabs\tand\r\nCRLF\n\n\n   trailing   ",
    "path/to/file.rs:42\n// comment\n/* block */", "fn main() {\n    println!(\"{}\", 3.14159);\n}\n",
    "<|start|>user<|message|>Hello<|end|><|start|>assistant<|channel|>final<|message|>Hi<|return|>",
    "<|startoftext|><|endoftext|> <|call|><|constrain|>json",
    "[gMASK]<sop><|system|>sys<|user|>hi<|assistant|><think>x</think><tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call><|observation|>",
    "Hello!  How's it going? 你好世界 12345678 don't", "根据 参考资料, 演员 王 而 者 据",  # glm4 ignore_merges words
    "    \n\t  nbsp em-space", "x" * 300 + " " + "9" * 31,
    "नमस्ते दुनिया ক্ষমা தமிழ் ภาษาไทย",  # \p{M} combining marks (qwen35 pre-tokenizer)
    "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call><|im_end|>",
]
out = []
for t in dict.fromkeys(texts):
    ids = tok.encode(t, add_special_tokens=False).ids
    out.append({"text": t, "parse_special": True, "tokens": ids})
dst = root / f"tokenize_{name}.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=0))
print("wrote", dst, len(out), "cases")
