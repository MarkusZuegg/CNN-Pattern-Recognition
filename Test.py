from typing import Any
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
import torchmetrics
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
# from pl_bolts.datamodules import CIFAR10DataModule

seed_everything(7)
BATCH_SIZE = 256

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(pl.LightningModule):
    def __init__(self, block, num_blocks, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.classes = num_classes
        self.in_planes = 64
        self.lr = lr

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", 
                                                             num_classes=self.classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        output = self.linear(out)
        return output
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 236 #45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        out = self(x)
        loss = F.cross_entropy(out,y)
        self.log('train_loss', loss,on_step=True,on_epoch=True)
        return loss 
    
    def evaluate(self, batch, stage=None):
        x,y = batch
        out = self(x)
        loss = F.cross_entropy(out,y)
        out = nn.Softmax(-1)(out) 
        logits = torch.argmax(out,dim=1)
        acc = self.accuracy(logits, y) #! maybe change this to github acc function     

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss, acc


    def test_step(self,batch,batch_idx):
        self.evaluate(batch, stage='test')
    
    def validation_step(self,batch,batch_idx):
        self.evaluate(batch, stage='val')

def ResNet18(lr=0.05):
    num_classes = 10
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, lr=lr)

class load_CIFAR10data(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        #! this could be where issue is
        self.train_transform = torchvision.transforms.Compose([ 
                                RandomCrop(32, padding=4),
                                RandomHorizontalFlip(),
                                ToTensor(),
                                cifar10_normalization()])
        self.test_transform = Compose([ToTensor(), cifar10_normalization()])

        self.train = torchvision.datasets.CIFAR10(root='./CIFAR10_data', download=True,
                                                train=True, transform=self.train_transform)
        self.test = torchvision.datasets.CIFAR10(root='./CIFAR10_data', download=True, 
                                                train=False, transform=self.test_transform)
        print('Data loaded')

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size)    

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.test, self.batch_size)
    
def main():
    max_epochs = 50
    data = load_CIFAR10data(BATCH_SIZE)
    #  data_new = CIFAR10_datamodule(batch_size)
    mod = ResNet18()
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1)
    trainer.fit(mod, data)
    trainer.test(mod, data)

if __name__ == '__main__': main()