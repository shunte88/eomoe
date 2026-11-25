#!/usr/bin/env python3
"""Reference dequantized rows from llama.cpp's gguf-py (needs numpy) -> tests/golden/dequant.json.

Usage: <python-with-numpy> tools/capture_dequant.py <model.gguf> <llama.cpp dir>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[2]) / "gguf-py"))
import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

# (tensor, row indices): one Q4_K and one Q6_K per kind of layout we load.
PICK = [
    ("token_embd.weight", [0, 151645]),          # Q4_K
    ("output.weight", [0, 7777]),                # Q6_K
    ("blk.0.ffn_down_exps.weight", [0, 100000]), # 3-D expert tensor, rows flattened
    ("blk.0.ffn_gate_exps.weight", [0, 90000]),  # Q4_K expert tensor
]

r = GGUFReader(sys.argv[1])
by_name = {t.name: t for t in r.tensors}
out = []
for name, rows in PICK:
    t = by_name[name]
    row_len = int(t.shape[0])
    data = np.asarray(t.data).reshape(-1)   # raw bytes of all rows
    row_bytes = data.size // (int(np.prod(t.shape)) // row_len)
    for ri in rows:
        raw = data[ri * row_bytes:(ri + 1) * row_bytes]
        vals = dequantize(raw, t.tensor_type).reshape(-1).astype(np.float32)
        assert vals.size == row_len
        out.append({"tensor": name, "type": t.tensor_type.name, "row": ri,
                    "values": [float(v) for v in vals]})
        print(name, t.tensor_type.name, ri, vals[:3])
dst = Path(__file__).resolve().parent.parent / "tests" / "golden" / "dequant.json"
dst.write_text(json.dumps(out))
print("wrote", dst, dst.stat().st_size, "bytes")
