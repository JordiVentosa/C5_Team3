import torch
from pathlib import Path

print("Testing imports...")
from models.baseline import Baseline, EncoderResNet18, DecoderGRU
from models.train_wrapper import CaptioningModule
from models.metrics import Metric
from datasets.vizwiz_dataset import VizWizDataset

print("✓ All imports successful")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

model = Baseline(device=device)
print("✓ Model initialized")

encoder = EncoderResNet18(device=device)
decoder = DecoderGRU(device=device)
print("✓ Encoder and Decoder initialized")

print("\nTesting forward pass...")
dummy_images = torch.randn(2, 3, 224, 224).to(device)
output = model(dummy_images)
print(f"✓ Forward pass successful. Output shape: {output.shape}")
assert output.shape[0] == 2 and output.shape[1] == 80 and output.shape[2] == 201

print("\nTesting training wrapper...")
module = CaptioningModule(model=model, device=device)
print("✓ CaptioningModule initialized")

dummy_captions = torch.randint(0, 80, (2, 201)).to(device)
metrics = module.training_step((dummy_images, dummy_captions))
print(f"✓ Training step successful. Loss: {metrics['loss']:.4f}")

print("\nTesting prediction...")
predictions = module.predict(dummy_images)
print(f"✓ Prediction successful. Generated {len(predictions)} captions")
print(f"Sample prediction: '{predictions[0][:50]}...'")

print("\nTesting metrics...")
metric = Metric()
test_preds = ["a dog on the grass", "a cat sleeping"]
test_refs = ["a dog playing on grass", "a cat is sleeping"]
scores = metric.compute(test_preds, test_refs)
print(f"✓ Metrics computed: {metric.format_metrics(scores)}")

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nThe pipeline is ready. You can now:")
print("1. Prepare your data in the ../data directory")
print("2. Run training with: python train.py --data_root ../data")
