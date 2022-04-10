import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelSummary, LearningRateMonitor, ModelCheckpoint

# ------------
# INIT GLOBAL
# ------------
pl.seed_everything(42)

MODEL_LIST = ['resnet18', 'vgg11_bn', 'efficientnet_b0']

DATASET_PATH = os.environ.get('PATH_DATASETS', 'data/')
CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'saved_models/ConvNets')

BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_model(model_name):
    assert model_name in MODEL_LIST
    print('-'*40)
    print(f'Create Model')
    print('-'*40)
    if model_name == 'resnet18':
        trained_model = torchvision.models.resnet18(pretrained=True)
        in_features = trained_model.fc.in_features
        trained_model.fc = nn.Linear(in_features, 10, bias=True)
        return trained_model

    elif model_name == 'vgg11_bn':
        trained_model = torchvision.models.vgg11_bn(pretrained=True)
        in_features = trained_model.classifier[0].in_features
        trained_model.classifier = nn.Sequential(
            nn.Linear(in_features, 200, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(200, 100, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(100, 10, bias=True),
        )
        return trained_model

    elif model_name == 'efficientnet_b0':
        trained_model = torchvision.models.efficientnet_b0(pretrained=True)
        in_features = trained_model.classifier[1].in_features
        trained_model.fc = nn.Linear(in_features, 10, bias=True)
        return trained_model

# Create CIFAR10Classifier


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, model_name, optimizer_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        return optimizer

    def _calculate_loss(self, batch, mode='train'):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        acc = ((y_hat.argmax(dim=-1)) == y).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='train')
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')


def train_model(model_name, save_name, trainer, train_loader, val_loader, **kargs):
    # Check whetere pretrained model exist
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + '.ckpt')
    # If yes, load it
    if os.path.isfile(pretrained_filename):
        print(f'Load pretrained model : {pretrained_filename}')
        model = CIFAR10Classifier.load_from_checkpoint(pretrained_filename)
    # else train it
    else:
        model = CIFAR10Classifier(model_name=model_name, **kargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best model after training
        model = CIFAR10Classifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model


def cli_main(args):
    # ------------
    # data : transform / dataset / dataloader
    # ------------
    model_name = args.model
    if args.save is None:
        save_name = model_name
    print(model_name)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Download dataset and get DATA_MEANS and DATA_STD
    print('-'*40)
    print(f'Dataset Download: {DATASET_PATH}')
    print('-'*40)
    dataset = CIFAR10(DATASET_PATH, train=True, download=True)
    data_mean = (dataset.data / 255.0).mean(axis=(0, 1, 2))
    data_std = (dataset.data / 255.0).std(axis=(0, 1, 2))

    # Create transform
    train_transform = T.Compose([
        T.RandomResizedCrop((32, 32)),
        T.ToTensor(),
        T.Normalize(data_mean, data_std)
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(data_mean, data_std)
    ])

    # Create dataset / dataloader
    train_dataset = CIFAR10(DATASET_PATH, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR10(DATASET_PATH, train=True, download=True, transform=test_transform)

    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ------------
    # train : trainer / train model
    # ------------

    TRAINER = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        gpus=1 if DEVICE == 'cuda' else None,
        max_epochs=5,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
            LearningRateMonitor('epoch'),
            EarlyStopping(monitor='val_acc', mode='max', patience=2),
            ModelSummary(),
        ],
        progress_bar_refresh_rate=1,
    )

    train_model(model_name, save_name, TRAINER, train_loader, val_loader,
                optimizer_hparams={"lr": 0.001, "weight_decay": 1e-4})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="モデル選択：['resnet18', 'vgg11_bn', 'efficientnet_b0']")
    parser.add_argument('-m', '--model', default='resnet18', type=str, choices=['resnet18', 'vgg11_bn', 'efficientnet_b0'])
    parser.add_argument('-s', '--save', default=None, type=str)
    args = parser.parse_args()
    
    cli_main(args)
