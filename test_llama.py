import os

from pathlib import Path
from models.llama.llama import Llama, Dialog


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

    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    print("\nResults:\n")
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}")
        print(
            f"{result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    main()
