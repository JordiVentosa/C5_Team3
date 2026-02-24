import torchvision
import torch
import wandb
from torch.utils.data import DataLoader
from Week1.src.utils.dataset import KittyDataset


def collate_fn(batch):
    return tuple(zip(*batch))


dataset = KittyDataset(root_dir="/home/msiau/workspace/jventosa/PostTFG/Master/C5_Team3/Week1/datasets/KITTI-MOTS")

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    params,
    lr=1e-4
)

for param in model.backbone.parameters():
    param.requires_grad = False
    
    from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 10

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

wandb.init(
    project="kitty-fasterrcnn",
    config={
        "learning_rate": 0.005,
        "batch_size": 4,
        "epochs": 10,
        "optimizer": "SGD"
    }
)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in tqdm(loader):

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        # Log batch losses
        wandb.log({
            "batch_loss": losses.item(),
            **{f"loss_{k}": v.item() for k, v in loss_dict.items()}
        })

    lr_scheduler.step()

    avg_loss = epoch_loss / len(loader)

    wandb.log({
        "epoch": epoch,
        "epoch_loss": avg_loss,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })

    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")