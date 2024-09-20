## Getting started

* Install dependencies

```bash
python3 -m pip install triton fairscale fire torch
```

* Download llama model

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
--local-dir $HOME/models/llama-3-8b-instruct \
--local-dir-use-symlinks False
```

* Clone repo and get branch

```bash
git clone https://github.com/vegaluisjose/kernels.git
cd kernels
git checkout dev
python3 test_llama.py
```
