import os
from argparse import Namespace

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from pytorch_lightning.callbacks import EarlyStopping, ModelSummary

class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def cli_main(result_dir='./RESULT', save_name='RESULT/MNIST', max_epochs=2, batch_size=16, num_workers=8, **model_kwargs):
    pl.seed_everything(1234)

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # ------------
    # model
    # ------------
    model = LitClassifier(**model_kwargs)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        default_root_dir=os.path.join(result_dir, save_name),
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[
            EarlyStopping(monitor='val_loss', mode='min', patience=2),
            ModelSummary(max_depth=2)
        ],
        progress_bar_refresh_rate=1,
    )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(dataloaders=test_loader)


if __name__ == '__main__':
    cli_main(result_dir=os.path.curdir, save_name='MNIST', max_epochs=1, batch_size=16, num_workers=4, hidden_dim=64, lr=1e-3)
    