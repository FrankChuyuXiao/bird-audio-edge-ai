
import torch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0

    for x, y in dataloader:

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss


def evaluate(model, dataloader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in dataloader:

            pred = model(x)

            predicted = pred.argmax(dim=1)

            correct += (predicted == y).sum().item()

            total += y.size(0)

    return correct / total
