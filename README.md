# Self Improving AI

This project contains a minimal example of a self-improving language model. It
continuously looks for new text files in `./data/incoming`. When new files are
found, their content is appended to a training dataset and the model performs a
brief fine-tuning step. The updated model is saved to `./model`.

The default base model is `Qwen/Qwen3-8B` from the Hugging Face hub. Because the
training loop runs in small increments, you can fine-tune and save your model
locally.

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

4. Run the script (use `--pretrained_model` to override the large default model):

```bash
python main.py --pretrained_model sshleifer/tiny-gpt2 --max_loops 0
```

For a quick smoke test without downloading large models, add `--dry_run` to skip
training entirely.

The `--max_loops 0` argument keeps the program running indefinitely. Each time a
new file appears in `data/incoming`, the contents will be used to further train
the model. Processed files are moved to `data/processed`.

While the script is running you can monitor training progress at
`http://localhost:7860`. The web page automatically updates with the latest
training loss as it becomes available.

To stop the program, press `Ctrl+C`.

## Notes

- The example uses `Qwen/Qwen3-8B`. This model is large, so you may need to
  experiment with batch size or quantization to fit it on your hardware.
- Training data is accumulated in `data/training_data.txt`. You can remove or
  edit this file to reset the training history.


## Docker

You can run the project inside a Docker container instead of installing Python
and dependencies locally.

Build the image:

```bash
docker build -t self-improving-ai .
```

Then run it, mounting your local `data` and `model` directories so the training
state persists between runs:

```bash
docker run -it --rm -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model:/app/model \
  self-improving-ai
```

The container launches `python main.py --pretrained_model sshleifer/tiny-gpt2 --max_loops 0` by default and exposes
the Gradio interface on port 7860.
Use `--dry_run` with the container command if you only want to verify that the
loop logic works without performing any training.
