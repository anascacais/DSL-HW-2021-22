import argparse
from cv2 import ml_LogisticRegression

import torch
from torch import nn
from torch.utils.data import DataLoader

from hw2_linear_crf import LinearChainCRF
from utils import configure_seed, configure_device, pairwise_features, OCRDataset, collate_samples, plot

import time


class LinearSequenceModel(nn.Module):
    def __init__(self, n_classes, n_features, pad_index=None, pad_value=None):
        super().__init__()
        self.pad_index = pad_index
        self.pad_value = pad_value
        # affine transformation to get emissions (aka unigram scores)
        self.affine = nn.Linear(n_features, n_classes)
        # bigram scores are inside the crf encoded as transitions
        self.crf = LinearChainCRF(n_classes, pad_index=pad_index)

    def get_mask(self, X):
        if self.pad_value is None:
            return None
        mask = X != self.pad_value
        return mask.all(dim=-1)

    def forward(self, X):
        return self.crf(self.affine(X), mask=self.get_mask(X))

    def loss(self, X, y):
        return self.crf.neg_log_likelihood(self.affine(X), y, mask=self.get_mask(X))


def train_batch(X, y, model, optimizer, gpu_id=None):
    X, y = X.to(gpu_id), y.to(gpu_id)
    model.train()
    optimizer.zero_grad()
    loss = model.loss(X, y)
    loss.backward() # autograd
    optimizer.step()
    return loss.item()


def evaluate(model, dataloader, gpu_id=None):
    model.eval()
    with torch.no_grad():
        y_hat = []
        y_true = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred,_ = model(x_batch)
            y_hat.extend([y_ for y in y_pred for y_ in y])  # y_pred is a list of lists
            y_true.extend(y_batch.squeeze().flatten().tolist())
        y_hat = torch.tensor(y_hat)
        y_true = torch.tensor(y_true)
        acc = torch.mean((y_hat == y_true).float()).item()
        return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='letter.data',
                        help="Path to letter.data OCR corpus.")
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-learning_rate', type=float, default=.001)
    parser.add_argument('-l2_decay', type=float, default=0.)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-gpu_id', type=int, default=None)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument("-no_pairwise", default=True, action="store_true",
                        help="""If you pass this flag, the model will use
                        binary pixel features instead of pairwise ones.""")
    opt = parser.parse_args()

    configure_seed(opt.seed)
    configure_device(opt.gpu_id)

    print("Loading data...")
    feature_function = pairwise_features if not opt.no_pairwise else None
    train_dataset = OCRDataset(opt.data, 'train', feature_function=feature_function)
    dev_dataset = OCRDataset(opt.data, 'dev', train_dataset.labels, feature_function=feature_function)
    test_dataset = OCRDataset(opt.data, 'test', train_dataset.labels, feature_function=feature_function)

    # ideally we would use a batch size larger than 1, but for the sake of simplicity
    # you can use equal to 1 so you don't need to deal with padding and masking
    if opt.batch_size > 1:
        print("Batch size > 1. You'll need to handle pad positions in you model.")
        n_classes = len(train_dataset.labels) + 1  # 27 (add 1 for pad)
        pad_index = 26
        pad_value = -1
        collate_fn = collate_samples
    else:
        n_classes = len(train_dataset.labels)  # 26
        pad_index = None
        pad_value = None
        collate_fn = None

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # define the model
    n_features = len(train_dataset.X[0][0])
    model = LinearSequenceModel(n_classes, n_features, pad_value=pad_value, pad_index=pad_index)
    model = model.to(opt.gpu_id)

    # get an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    train_mean_losses = []
    valid_accs = []
    train_losses = []
    start_time = time.time()
    for ii in range(1, opt.epochs + 1):
        print('\nTraining epoch {}'.format(ii))
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            print('{} of {}'.format(i + 1, len(train_dataloader)), end='\r')
            loss = train_batch(X_batch, y_batch, model, optimizer, gpu_id=opt.gpu_id)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % mean_loss)

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_dataloader, gpu_id=opt.gpu_id))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    run_time = time.time() - start_time
    print(f'Training time: {run_time}')

    print('Final Test acc: %.4f' % (evaluate(model, test_dataloader, gpu_id=opt.gpu_id)))
    # plot
    str_epochs = [str(i) for i in range(1, opt.epochs + 1)]
    plot(str_epochs, train_mean_losses, ylabel='Loss', name='crf-training-loss')
    plot(str_epochs, valid_accs, ylabel='Accuracy', name='crf-validation-accuracy')


if __name__ == "__main__":
    main()
