"""
Script for preparing the Tulu V2 data for fine-tuning an OLMo model.
"""

import logging
import random
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)

    dataset = ds.load_dataset(opts.datapath, split=opts.split)

    log.info("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess,
            tokenizer=tokenizer,
            max_seq_len=opts.seq_len,
            with_ready_pause=opts.use_rptok,
            span_probability=opts.probability,
            mean_span_length=opts.length,
            ),
        batched=False,
        remove_columns=["dataset", "id", "messages"],
        num_proc=opts.num_proc,  # type: ignore
    )

    log.info("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)  # type: ignore
    log.info(f"Filtered out {n - len(dataset):,d} examples")

    log.info("Counting tokens...")
    total_tokens = 0
    for ex in track(dataset):
        assert len(ex["input_ids"]) == opts.seq_len  # type: ignore
        total_tokens += len(ex["input_ids"])  # type: ignore
    log.info(f"Total tokens: {total_tokens:,d}")

    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"), dtype=np.uint16, mode="w+", shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(total_tokens,)
    )
    offset = 0
    for ex in track(dataset):
        ex_len = len(ex["input_ids"])  # type: ignore
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]  # type: ignore
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]  # type: ignore
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def preprocess(
    example,
    tokenizer: Tokenizer,
    max_seq_len: int,
    with_ready_pause: bool = False,
    span_probability: float = 0.50,
    mean_span_length: int = 100,
    pause_token: str = "<|pause|>",
    ready_token: str = "<|ready|>",
    ):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    if with_ready_pause:
        pause_id = tokenizer.encode(pause_token, add_special_tokens=False)[0]
        ready_id = tokenizer.encode(ready_token, add_special_tokens=False)[0]

    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":

            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
            
            # Inject sequences of ready pause tokens
            if with_ready_pause:
                content_length = len(content_tokens)

                if random.random() < span_probability:
                    # think_length = np.random.randint(1, mean_span_length)
                    think_length = int(np.random.normal(mean_span_length, mean_span_length/3, 1)[0])
                    think_span = [pause_id]*(think_length - 1)+[ready_id]
                    content_tokens = np.concatenate([think_span, content_tokens]).tolist()

            label_mask += [True] * len(content_tokens)
            # mask out the last '\n'
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Tulu V2 dataset")
    parser.add_argument("output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default=Path(__file__).parent / "tokenizers" / "allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
    )
    parser.add_argument("-s", "--seq-len", type=int, help="""Max sequence length.""", default=2048)
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=50279)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=1)
    parser.add_argument("-j", "--num-proc", type=int, help="""Number of workers.""", default=8)
    parser.add_argument("--use_rptok", action='store_true', help="""Add pause ready tokens""")
    parser.add_argument("-p", "--probability", type=float, help="""Probability of using rptok""", default=0.5)
    parser.add_argument("-l", "--length", type=float, help="""Max length of rptok""", default=50.0)
    parser.add_argument("--datapath", type=str, help="""Huggingface Datset""")
    parser.add_argument("--split", type=str, help="""Dataset Split""", default="train")
    
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)
