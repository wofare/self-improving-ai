import argparse
import os
import time
import threading
# Heavy dependencies are imported lazily to allow a lightweight dry run


# Shared progress log accessible by the training loop and the web UI
progress_log = []




def main():
    parser = argparse.ArgumentParser(description="Self-improving language model")
    parser.add_argument('--model_path', default='./model', help='Where to save or load the model')
    parser.add_argument('--data_dir', default='./data', help='Directory with incoming data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--steps_per_loop', type=int, default=100)
    parser.add_argument('--sleep_secs', type=int, default=10)
    parser.add_argument('--max_loops', type=int, default=1, help='Number of improvement loops (0 for infinite)')
    parser.add_argument('--pretrained_model', default='Qwen/Qwen3-8B',
                        help='HF model to use when no local model is found')
    parser.add_argument('--dry_run', action='store_true',
                        help='Skip model downloads and training')
    args = parser.parse_args()

    if args.dry_run:
        print('Running in dry run mode; skipping model loading and training.')
        incoming_dir = os.path.join(args.data_dir, 'incoming')
        processed_dir = os.path.join(args.data_dir, 'processed')
        os.makedirs(incoming_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        loop = 0
        while args.max_loops == 0 or loop < args.max_loops:
            incoming_files = [f for f in os.listdir(incoming_dir) if f.endswith('.txt')]
            if not incoming_files:
                print('No new data found. Sleeping...')
                time.sleep(args.sleep_secs)
                loop += 1
                continue
            print(f'Found {len(incoming_files)} new file(s). Processing (dry run)...')
            for fname in incoming_files:
                path = os.path.join(incoming_dir, fname)
                os.rename(path, os.path.join(processed_dir, fname))
            progress_log.append({'step': loop, 'loss': 'dry_run'})
            print('Dry run iteration completed. Waiting for new data...')
            time.sleep(args.sleep_secs)
            loop += 1
        return



    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )
    import gradio as gr
    from datasets import load_dataset as hf_load_dataset

    class ProgressCallback(TrainerCallback):
        """Collect training metrics for display in the web UI."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                entry = {"step": state.global_step}
                entry.update(logs)
                progress_log.append(entry)

    def launch_ui():
        """Start the Gradio web UI in a separate thread."""
        with gr.Blocks() as demo:
            log_md = gr.Markdown()

            def update():
                if not progress_log:
                    return "Waiting for training to start..."
                lines = [
                    f"**Step {p['step']}** - loss: {p.get('loss', 'N/A')}" for p in progress_log[-20:]
                ]
                return "\n".join(lines)

            demo.load(update, None, log_md, every=1)

        demo.launch(server_name="0.0.0.0", share=False)

    def load_local_dataset(file_path: str, tokenizer: AutoTokenizer):
        """Load and tokenize text from ``file_path``."""
        ds = hf_load_dataset('text', data_files={'train': file_path})['train']

        def tokenize(batch):
            return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

        ds = ds.map(tokenize, batched=True)
        ds = ds.remove_columns(['text'])
        ds.set_format(type='torch')
        return ds

    # Start the web UI in a background thread
    threading.Thread(target=launch_ui, daemon=True).start()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_files_exist = os.path.isfile(os.path.join(args.model_path, 'config.json'))
    if model_files_exist:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)

    incoming_dir = os.path.join(args.data_dir, 'incoming')
    processed_dir = os.path.join(args.data_dir, 'processed')
    training_file = os.path.join(args.data_dir, 'training_data.txt')

    os.makedirs(incoming_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    open(training_file, 'a').close()

    loop = 0
    while args.max_loops == 0 or loop < args.max_loops:
        incoming_files = [f for f in os.listdir(incoming_dir) if f.endswith('.txt')]
        if not incoming_files:
            print('No new data found. Sleeping...')
            time.sleep(args.sleep_secs)
            loop += 1
            continue

        print(f'Found {len(incoming_files)} new file(s). Adding to dataset and training...')
        with open(training_file, 'a', encoding='utf-8') as train_f:
            for fname in incoming_files:
                path = os.path.join(incoming_dir, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    train_f.write(f.read())
                    train_f.write("\n")
                os.rename(path, os.path.join(processed_dir, fname))

        dataset = load_local_dataset(training_file, tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=args.model_path,
            overwrite_output_dir=True,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=1,
            save_steps=args.steps_per_loop,
            logging_steps=10,
            logging_dir=os.path.join(args.model_path, "logs"),
            no_cuda=not torch.cuda.is_available(),
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback()],
        )

        trainer.train()
        trainer.save_model(args.model_path)
        tokenizer.save_pretrained(args.model_path)
        print('Training iteration completed. Waiting for new data...')
        time.sleep(args.sleep_secs)
        loop += 1


if __name__ == '__main__':
    main()
