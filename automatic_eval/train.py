from torch import optim


def evaluate(dataloader, model, args):
    for batch in dataloader:
        inputs, label = batch
        logits, _ = model(**inputs)
        loss = model.calc_loss(logits, label)

    return loss_summary


def train(dataloaders, model, args):
    train_loader, valid_loader, test_loader = dataloaders
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for i in range(args.epochs):
        for batch in train_loader:
            inputs, label = batch
            logits, _ = model(**inputs)

            optimizer.zero_grad()
            loss = model.calc_loss(logits, label)
            model.backward(loss)
            optimizer.step()

        # evaluation
        valid_loss_summary = evaluate(valid_loader, model, args)
        test_loss_summary = evaluate(test_loader, model, args)
