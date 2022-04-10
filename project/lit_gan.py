import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from pytorch_lightning.callbacks import ModelCheckpoint


DATASET_PATH = os.environ.get('PATH_DATASETS', 'data/')
CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'saved_models/GAN')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() * 2)

# Define a DataModule for CIFAL10 dataset
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transformer
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                (0.1307,),
                (0.3081,)
            )
        ])
        self.dims = (1,28,28)
        self.num_classes=10
      
    def prepare_data(self):
        # download
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign each datasets for use in dataloaders
        if stage in ['fit', None]:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        
        if stage in ['test', None]:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True          
        )

    def validation_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True          
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers        
        )

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256), 
            *block(256, 512), 
            *block(512, 1024), 
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
      img_flat = img.view(img.size(0), -1)
      validity = self.model(img_flat)
      return validity

class SimpleGAN(pl.LightningModule):
    def __init__(
        self,
        channels, width, height,
        latent_dim = 100,
        lr = 2e-4,
        b1 = 0.5,
        b2 = 0.999,
        batch_size = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        ## Init networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim, data_shape)
        self.discriminator = Discriminator(data_shape)

        # For sample imgs
        self.validations_z = torch.rand(8, latent_dim)
        self.example_input_array = torch.zeros(2, latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        
        z = torch.rand(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # Train Generator
        if optimizer_idx == 0:
            # Generate images
            self.generated_imgs = self(z)

            # For log images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images_1', grid, self.current_epoch)

            # Make ground truth result (REAK : 1)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # Calculate adversarial loss
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss':g_loss}
            # output = OrderedDict({'loss':g_loss, 'progress_bar':tqdm_dict, 'log':tqdm_dict})

            self.log('g_loss', g_loss, prog_bar=True, on_epoch=True)
            # return output

        # Train Discriminator
        if optimizer_idx == 1:
            # Calculate loss for real images
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # Calculate loss for fake images
            valid = torch.zeros(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), valid)

            # Discriminator loss is the average of real / fake loss
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            # output = OrderedDict({'loss':d_loss, 'progress_bar':tqdm_dict,'log':tqdm_dict})

            self.log('d_loss', d_loss, prog_bar=True, on_epoch=True)
            # return output

    # 2 Optimizers
    def configure_optimizers(self):
        lr, b1, b2 = self.hparams.lr, self.hparams.b1, self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr, betas=(b1,b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr, betas=(b1,b2))
        return [opt_g, opt_d]

    def on_epoch_end(self):
        z = self.validations_z.type_as(self.generator.model[0].weight)
        # log images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

if __name__ == '__main__':
    dm = MNISTDataModule(DATASET_PATH, BATCH_SIZE, NUM_WORKERS)
    model = SimpleGAN(*dm.size())
    trainer = pl.Trainer(
        default_root_dir=CHECKPOINT_PATH, 
        gpus=AVAIL_GPUS, 
        max_epochs=5, 
        progress_bar_refresh_rate=5,
        callbacks=[
            ModelCheckpoint(filename='sample-test-{epoch:02d}',
                            monitor='g_loss',
                            save_weights_only=True,
                            )
        ],
    )
    trainer.fit(model, dm)
    model = SimpleGAN(*dm.size()).load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    # Convert to onnx
    t_batch = next(iter(dm.test_dataloader()))
    t_input = t_batch['input_ids'][:1]

    print('-'*20, '\nCONVERT TO ONNX\n', '-'*20)
    save_path = os.path.join(CHECKPOINT_PATH,'little_GAN.onnx')
    print(f'SAVE INTO : {save_path}"')
    model.to_onnx(save_path, t_input, export_params=True, opset_version=12)
    
    