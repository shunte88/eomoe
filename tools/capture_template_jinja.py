#!/usr/bin/env python3
"""Chat-template goldens from Python jinja2, configured as HF transformers' apply_chat_template
(trim_blocks, lstrip_blocks, loopcontrols, tojson = json.dumps(ensure_ascii=False),
raise_exception), with the template read from the GGUF -> tests/golden/template_<arch>.json.
Like llama.cpp, tool-call arguments are handed to the template parsed. strftime_now renders a
placeholder "<DATE:fmt>" the gate fills with today's date.

Usage: python3 tools/capture_template_jinja.py <model.gguf>   (needs jinja2)
"""
import json, struct, sys, copy
from pathlib import Path
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment


def gguf_kv(path):
    f = open(path, "rb")
    r = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    s = lambda: f.read(r("Q")).decode()
    F = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def val(t):
        if t == 8: return s()
        if t == 9:
            et, n = r("I"), r("Q")
            return [val(et) for _ in range(n)]
        return r(F[t])
    assert f.read(4) == b"GGUF"
    _, _, nkv = r("I"), r("Q"), r("Q")
    kv = {}
    for _ in range(nkv):
        k = s(); kv[k] = val(r("I"))
        if k == "tokenizer.chat_template": return kv


def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)


kv = gguf_kv(sys.argv[1])
arch = kv["general.architecture"]
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = lambda fmt: f"<DATE:{fmt}>"
tpl = env.from_string(kv["tokenizer.chat_template"])

root = Path(__file__).resolve().parent.parent / "tests" / "golden"
reqs = [c["request"] for c in json.load(open(root / "template.json"))]
reqs += [
    {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "thinking": "Easy: 4.", "content": "4"}, {"role": "user", "content": "And 3+3?"}], "add_generation_prompt": True},
    {"messages": [{"role": "developer", "content": "Answer in French."}, {"role": "user", "content": "Hi"}], "add_generation_prompt": False},
    {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "reasoning_content": "Easy: 4.", "content": "4"}, {"role": "user", "content": "And 3+3?"}, {"role": "assistant", "reasoning_content": "Six.", "content": "6"}], "add_generation_prompt": False},
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "\n<think>\n Plan: greet. \n</think>\n\n  Hello there!  "}], "add_generation_prompt": False},
    {"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"enable_thinking": False}},
]
out = []
for req in reqs:
    ctx = copy.deepcopy(req)
    for m in ctx["messages"]:
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                try: tc["function"]["arguments"] = json.loads(a)
                except ValueError: pass
    ctx.setdefault("add_generation_prompt", True)
    ctx.update(ctx.pop("chat_template_kwargs", {}))  # as llama-server passes them
    try:
        out.append({"request": req, "prompt": tpl.render(**ctx)})
    except (TypeError, jinja2.exceptions.TemplateError) as e:  # the template's own bug: callers get an error
        out.append({"request": req, "error": f"{type(e).__name__}: {e}"})
dst = root / f"template_{arch}.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("wrote", dst, len(out), "cases")
