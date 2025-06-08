# Self Improving AI

This project contains a minimal example of a self-improving language model. It
continuously looks for new text files in `./data/incoming`. When new files are
found, their content is appended to a training dataset and the model performs a
brief fine-tuning step. The updated model is saved to `./model`.

The default base model is `gpt2` from the Hugging Face hub. Because the training
loop runs in small increments and the model is lightweight, the script can be
run locally on a single RTX 3080 GPU.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare directories:

```bash
mkdir -p data/incoming data/processed
```

3. Place any `.txt` files you want the model to learn from in `data/incoming`.

4. Run the script:

```bash
python main.py --max_loops 0
```

The `--max_loops 0` argument keeps the program running indefinitely. Each time a
new file appears in `data/incoming`, the contents will be used to further train
the model. Processed files are moved to `data/processed`.

To stop the program, press `Ctrl+C`.

## Notes

- The example uses `gpt2`, which is about 124M parameters. Larger models may
  require more VRAM.
- Training data is accumulated in `data/training_data.txt`. You can remove or
  edit this file to reset the training history.

