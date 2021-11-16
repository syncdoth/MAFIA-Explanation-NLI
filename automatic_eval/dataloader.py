from torch.utils.data import Dataset, DataLoader


def get_dataloaders(tokenizer, args):
    train_loader = DataLoader(NLIDataset(tokenizer, 'train', args),
                              args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(NLIDataset(tokenizer, 'valid', args),
                              args.eval_batch_size,
                              shuffle=False)
    test_loader = DataLoader(NLIDataset(tokenizer, 'test', args),
                             args.eval_batch_size,
                             shuffle=False)
    return train_loader, valid_loader, test_loader


class NLIDataset(Dataset):

    def __init__(self, tokenizer, mode, args):
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
