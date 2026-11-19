"""Try a trained model: type a story opening, the model continues it.

Usage:
    python scripts/sample.py                      # interactive, largest model (S5)
    python scripts/sample.py --model s3           # a smaller model, to compare
    python scripts/sample.py --prompt "Once upon a time" --n 3
"""

from __future__ import annotations

import argparse
import glob

import torch

from tinygpt.model.generate import generate
from tinygpt.model.gpt import load_gpt
from tinygpt.tokenizer.bpe import BPETokenizer
from tinygpt.train.metrics import pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="s5", help="s1 (smallest) ... s5 (largest)")
    ap.add_argument("--prompt", default=None, help="if omitted, asks interactively")
    ap.add_argument("--n", type=int, default=1, help="continuations per prompt")
    ap.add_argument("--tokens", type=int, default=150, help="max new tokens")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy, higher = wilder")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tokenizer", default="data/tokenizer/tinystories_8k.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"runs/sweeps/scaling/*_{args.model}_*/ckpt_final.pt"))
    if not paths:
        raise SystemExit(f"no finished checkpoint for {args.model!r}")
    device = pick_device()
    model, _ = load_gpt(paths[-1], device)
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    eot = tok.encode("<|endoftext|>", allowed_special=True)[0]
    print(f"Loaded {args.model} from {paths[-1]} on {device}.")

    def run(prompt: str) -> None:
        ids = torch.tensor([tok.encode(prompt)], device=device)
        for i in range(args.n):
            out = generate(
                model, ids, args.tokens, temperature=args.temperature, top_k=args.top_k, eos_id=eot
            )
            text = tok.decode(out[0].tolist()).replace("<|endoftext|>", "")
            print(f"\n--- {i + 1} ---\n{text}\n")

    if args.prompt is not None:
        run(args.prompt)
        return
    print("Type a story opening and press Enter (empty line or Ctrl+C to quit).")
    try:
        while prompt := input("\nprompt> ").strip():
            run(prompt)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
