#!/usr/bin/env python3
"""Export Underdog Woof for the Rust port, keeping the int8 weights int8.

The npz already stores int8 (`q`/`q8`) plus scales — the numpy runtime dequantises all of it to
f32 at load, which is where its ~205 MB of resident weights comes from. Here we keep the quantised
form and record just enough metadata for Rust to reproduce the same dequantisation:

  speech linears   {"q": i8 (out,in), "s": f32 (out,)}        -> w[o,i] = q[o,i] * s[o]
  decoder tensors   {"q8": i8 (rows,n), "s": f32 (rows,ng),
                     "shape", "axis", "g"}                   -> grouped-int8, see `q8meta`

    python export.py --repo <underdog_woof checkout> --out /tmp/woof-i8
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--f32", action="store_true", help="dequantise instead (old behaviour, for A/B)")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from underdog_woof.runtime import unflatten  # noqa: E402

    with np.load(os.path.join(a.repo, "woof_int8.npz")) as z:
        tree = unflatten(z)

    entries = {}
    meta = {}

    def put(name, arr, dtype="f32"):
        if arr is None:
            return
        assert name not in entries, name
        arr = np.ascontiguousarray(np.asarray(arr))
        entries[name] = arr.astype(np.float32) if dtype == "f32" else arr.astype(np.int8)
        meta[name] = {"dtype": dtype, "shape": list(arr.shape)}

    def put_lin(name, node):
        if not a.f32 and "q" in node:
            put(name + ".q", node["q"], "i8")
            put(name + ".s", node["s"], "f32")
        else:
            w = node["w"] if "w" in node else node["q"].astype(np.float32) * node["s"][:, None]
            put(name + ".w", w, "f32")
        if "b" in node:
            put(name + ".b", node["b"], "f32")

    def put_q8(name, node):
        if not a.f32 and "q8" in node:
            put(name + ".q", node["q8"], "i8")
            put(name + ".s", node["s"], "f32")
            meta[name] = {"dtype": "q8", "shape": [int(v) for v in node["shape"]],
                          "axis": int(node["axis"]), "g": int(node["g"]),
                          "rows": int(node["q8"].shape[0]), "n": int(node["q8"].shape[1]),
                          "ng": int(node["s"].shape[1])}
        else:
            from underdog_woof.runtime import deq
            put(name, deq(node), "f32")

    # ---- speech -------------------------------------------------------------------------
    sp = tree["speech"]
    for i, layer in enumerate(sp["layers"]):
        for k, node in enumerate(layer):
            put_lin(f"sp.layers.{i}.{k}", node)
    for nm in ("ce", "ada", "lat_in", "h_inp", "h_out"):
        put_lin(f"sp.{nm}", sp[nm])
    for j, blk in enumerate(sp["hblocks"]):
        for k, node in enumerate(blk):
            put_lin(f"sp.hblocks.{j}.{k}", node)
    put("sp.text_emb", sp["text_emb"])
    put("sp.bos", sp["bos"])
    for v in ("juno", "mira"):
        put(f"sp.voice.{v}", tree["voices"][v])
    put("mean", tree["mean"])
    put("std", tree["std"])

    # ---- decoder ------------------------------------------------------------------------
    dec = tree["decoder"]
    put("dec.inp_w", dec["inp"][0].astype(np.float32).T)
    put("dec.inp_b", dec["inp"][1])
    put("dec.inv_freq", dec["inv_freq"])
    for i, l in enumerate(dec["tf"]):
        for nm in ("qkv", "o", "fc1", "fc2"):
            put_q8(f"dec.tf.{i}.{nm}", l[nm])
        for ln in ("ln1", "ln2"):
            put(f"dec.tf.{i}.{ln}.0", l[ln][0])
            put(f"dec.tf.{i}.{ln}.1", l[ln][1])

    seanet = []
    for idx, (kind, p) in enumerate(dec["seanet"]):
        if kind == "res":
            rec = {"kind": kind, "sub": []}
            for sub_i, sub in enumerate(p):
                put_q8(f"dec.seanet.{idx}.{sub_i}.w", sub["w"])
                put(f"dec.seanet.{idx}.{sub_i}.b", sub["b"])
                shape = meta[f"dec.seanet.{idx}.{sub_i}.w"]["shape"] if f"dec.seanet.{idx}.{sub_i}.w" in meta else None
                rec["sub"].append({"cin": (shape[1] if shape and "q8" not in meta.get(f"dec.seanet.{idx}.{sub_i}.w", {}) else None)})
            seanet.append(rec)
        elif kind in ("conv", "convt"):
            put_q8(f"dec.seanet.{idx}.w", p["w"])
            put(f"dec.seanet.{idx}.b", p["b"])
            rec = {"kind": kind}
            seanet.append(rec)
        else:
            seanet.append({"kind": kind})

    # shapes of the seanet weights (needed for taps / convt reshape) come from the npz metadata
    for idx, (kind, p) in enumerate(dec["seanet"]):
        if kind == "conv":
            m = meta[f"dec.seanet.{idx}.w"]
            cout, cin, k = m["shape"]
            seanet[idx].update({"k": k, "cin": cin, "cout": cout})
        elif kind == "convt":
            m = meta[f"dec.seanet.{idx}.w"]
            cin, cout, k = m["shape"]
            seanet[idx].update({"s": int(p["stride"]), "kin": k, "cin": cin, "cout": cout})
        elif kind == "res":
            for sub_i, sub in enumerate(p):
                m = meta[f"dec.seanet.{idx}.{sub_i}.w"]
                cout, cin, k = m["shape"]
                seanet[idx]["sub"][sub_i] = {"k": k, "cin": cin, "cout": cout}

    cfg = {
        "sr": 24000,
        "sp": {"L": len(sp["layers"]), "d": int(sp["cfg"]["d"]), "H": int(sp["cfg"]["H"]),
               "ff": int(sp["cfg"]["ff"]), "hw": int(sp["cfg"]["hw"]), "depth": int(sp["cfg"]["depth"]),
               "latent": int(sp["cfg"]["latent"]), "Dh": int(sp["cfg"]["d"]) // int(sp["cfg"]["H"]),
               "max_len": 1100},
        "dec": {"nh": int(dec["cfg"]["heads"]), "hd": int(dec["cfg"]["head_dim"]),
                "window": int(dec["cfg"]["window"]), "eps": float(dec["cfg"]["eps"]),
                "chunk": 8, "split": 6,
                "D": int(dec["cfg"]["heads"]) * int(dec["cfg"]["head_dim"])},
        "seanet": seanet,
    }

    offset = 0
    tensors = {}
    with open(a.out + ".bin", "wb") as f:
        for name, arr in entries.items():
            data = arr.tobytes()
            f.write(data)
            m = dict(meta[name])
            m.update({"offset": offset, "count": int(arr.size)})
            tensors[name] = m
            offset += len(data)
    # grouped-int8 tensors have metadata but no payload of their own (payload is "<name>.q")
    for name, m in meta.items():
        if m.get("dtype") == "q8":
            tensors[name] = dict(m)
    with open(a.out + ".json", "w") as f:
        json.dump({"cfg": cfg, "tensors": tensors}, f)

    i8 = sum(m["count"] for m in tensors.values() if m["dtype"] == "i8")
    f32 = sum(m["count"] for m in tensors.values() if m["dtype"] == "f32")
    print(f"{len(tensors)} tensors: {i8/2**20:.1f} MiB int8 + {f32/2**20:.1f} MiB f32 = {offset/2**20:.1f} MiB "
          f"-> {a.out}.bin")
    print("speech cfg:", cfg["sp"])
    print("decoder cfg:", cfg["dec"])


if __name__ == "__main__":
    main()
