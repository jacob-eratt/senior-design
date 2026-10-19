"""Train the float ResNet8 on CIFAR-10.

    python -m resnet8.train --epochs 40

Augmentation (random 4-pixel-padded crop + horizontal flip) runs on whole
in-memory batches, so training needs no DataLoader workers and runs at a
reasonable speed on a laptop CPU.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import load_cifar10, to_float
from .model import ResNet8

PACKAGE_DIR = Path(__file__).resolve().parent


def augment(batch, generator):
    """Random crop with 4-pixel zero padding, then random horizontal flip."""
    n, c, h, w = batch.shape
    padded = nn.functional.pad(batch, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (n,), generator=generator)
    ox = torch.randint(0, 9, (n,), generator=generator)
    rows = (oy[:, None] + torch.arange(h))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w))[:, None, None, :]
    cropped = padded[torch.arange(n)[:, None, None, None], torch.arange(c)[None, :, None, None], rows, cols]
    flip = torch.rand(n, generator=generator) < 0.5
    cropped[flip] = cropped[flip].flip(3)
    return cropped


@torch.inference_mode()
def evaluate(model, images_u8, labels, batch_size=1000):
    model.eval()
    correct = 0
    for start in range(0, len(labels), batch_size):
        logits = model(to_float(images_u8[start:start + batch_size]))
        correct += (logits.argmax(1).numpy() == labels[start:start + batch_size]).sum()
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "resnet8.pth")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    train_x, train_y = load_cifar10(train=True)
    test_x, test_y = load_cifar10(train=False)
    train_x_f = to_float(train_x)
    train_y_t = torch.as_tensor(train_y)

    model = ResNet8()
    steps_per_epoch = len(train_y) // args.batch_size
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.15,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        start = time.perf_counter()
        order = torch.randperm(len(train_y), generator=generator)
        total_loss, correct = 0.0, 0
        for step in range(steps_per_epoch):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            x = augment(train_x_f[idx], generator)
            y = train_y_t[idx]
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
        seen = steps_per_epoch * args.batch_size
        test_acc = evaluate(model, test_x, test_y)
        print(
            f"epoch {epoch + 1:3d}/{args.epochs} | loss {total_loss / steps_per_epoch:.4f} | "
            f"train {correct / seen:.2%} | test {test_acc:.2%} | {time.perf_counter() - start:.0f}s",
            flush=True,
        )

    torch.save(model.state_dict(), args.output)
    print(f"Saved {args.output} (test accuracy {evaluate(model, test_x, test_y):.2%})")


if __name__ == "__main__":
    np.set_printoptions(linewidth=120)
    main()
