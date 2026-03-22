import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm

from models.baseline import Baseline
from models.train_wrapper import CaptioningModule
from custom_datasets.vizwiz_dataset import VizWizDataset
from text_tokenizers import get_tokenizer

# Global variables for configuration
DATA_ROOT = None
RESNET_MODEL = None
BATCH_SIZE = None
NUM_WORKERS = None
RNN_TYPE = None
DEVICE = None


def run_inference(module, dataloader):
    """Run inference on the test set and compute metrics."""
    all_predictions = []
    all_references = []
    progress_bar = tqdm(dataloader, desc="Running inference on test set")

    for batch in progress_bar:
        images, captions, caption_texts = batch

        # Generate predictions
        predictions = module.predict(images)
        all_predictions.extend(predictions)
        all_references.extend(caption_texts)

    # Compute metrics
    metrics = module.compute_metrics(all_predictions, all_references)

    return all_predictions, all_references, metrics


def main(args):
    global DATA_ROOT, RESNET_MODEL, BATCH_SIZE, NUM_WORKERS, RNN_TYPE, DEVICE

    # Set global variables from args
    DATA_ROOT = args.data_root
    RESNET_MODEL = args.resnet_model
    RNN_TYPE = args.rnn_type
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Using configuration:")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"RESNET_MODEL: {RESNET_MODEL}")
    print(f"RNN_TYPE: {RNN_TYPE}")
    print(f"TOKENIZER_TYPE: {args.tokenizer_type}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"DEVICE: {DEVICE}")
    print(f"CHECKPOINT: {args.checkpoint}\n")

    # Initialize tokenizer
    print(f"Initializing {args.tokenizer_type} tokenizer...")
    tokenizer = get_tokenizer(tokenizer_type=args.tokenizer_type)

    # Load training dataset to build vocabulary (same as during training)
    print("Loading training dataset to build vocabulary...")
    train_dataset_temp = VizWizDataset(data_root=Path(DATA_ROOT), split='train', tokenizer=None)
    train_captions = train_dataset_temp.get_all_captions()
    tokenizer.build_vocab(train_captions)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Max sequence length: {tokenizer.max_len}")

    print("\nLoading test dataset...")
    # Test set is the original validation set
    test_dataset = VizWizDataset(data_root=Path(DATA_ROOT), split='val', tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))

    print(f"Test: {len(test_dataset)} samples")

    print("\nCreating model...")
    model = Baseline(
        tokenizer=tokenizer,
        device=DEVICE,
        resnet_model=RESNET_MODEL,
        rnn_type=RNN_TYPE,
        freeze_encoder=False  # Not needed for inference, but keeping signature consistent
    )
    module = CaptioningModule(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE
    )

    print(f"Loading checkpoint from {args.checkpoint}...")
    if Path(args.checkpoint).exists():
        module.load_checkpoint(args.checkpoint)
        print("✓ Model loaded successfully\n")
    else:
        print(f"⚠ Checkpoint not found: {args.checkpoint}")
        print("Aborting inference - checkpoint is required.")
        return

    print(f"{'='*80}\nRunning inference on test set...\n{'='*80}\n")

    predictions, references, metrics = run_inference(module, test_loader)

    print(f"\n{'='*80}")
    print("Test Results:")
    print(f"{'='*80}")
    print(f"Test Metrics: {module.metric.format_metrics(metrics)}")
    print(f"{'='*80}\n")

    if args.save_predictions:
        output_path = Path(args.output_dir) / "test_predictions.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for pred, ref in zip(predictions, references):
                f.write(f"Prediction: {pred}\n")
                f.write(f"Reference: {ref}\n")
                f.write("-" * 80 + "\n")

        print(f"✓ Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory of VizWiz dataset")
    parser.add_argument("--tokenizer_type", type=str, default="character",
                        choices=["character", "word", "subword"],
                        help="Tokenizer type: character, word, or subword (BERT)")
    parser.add_argument("--resnet_model", type=str, default="microsoft/resnet-18", help="ResNet model for encoder")
    parser.add_argument("--rnn_type", type=str, default="GRU", choices=["GRU", "LSTM"], help="RNN type for decoder")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for predictions")

    args = parser.parse_args()
    main(args)

