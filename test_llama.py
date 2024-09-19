import os

from pathlib import Path
from models.llama.llama import Llama, Dialog
from models.llama.llama.tokenizer import Tokenizer, ChatFormat
from time import perf_counter

import torch.distributed as dist


def main():
    ckpt_dir = Path.home().joinpath("models", "llama-3-8b-instruct", "original")
    tokenizer_path = ckpt_dir.joinpath("tokenizer.model")

    max_seq_len = 512
    max_batch_size = 4
    use_triton = True

    max_gen_len = None
    temperature = 0.6
    top_p = 0.9

    dialogs: List[Dialog] = [[{"role": "user", "content": "count to ten"}]]

    generator = Llama.build(
        ckpt_dir=str(ckpt_dir),
        tokenizer_path=str(tokenizer_path),
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        use_triton=use_triton,
    )

    tic = perf_counter()

    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    toc = perf_counter()

    formatter = ChatFormat(Tokenizer(str(tokenizer_path)))

    print("\nResults:\n")
    for dialog, result in zip(dialogs, results):
        tokens = formatter.encode_dialog_prompt(dialog=dialog)
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}")
        print(
            f"{result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print(f"\nMetrics:")
        print(f"Elapsed time (s): {toc-tic}")
        print(f"Throughput (tok/s): {len(tokens)/(toc-tic)}")


if __name__ == "__main__":
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    main()

    # fix the following warning:
    # Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL.
    if dist.is_initialized():
        dist.destroy_process_group()
