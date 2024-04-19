import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
import torchmetrics
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, Normalize
import pandas as pd
import seaborn as sn
from pylab import savefig
import matplotlib.pyplot as plt
from IPython.core.display import display
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

class BasicBlock(nn.Module):
    """Creates a block of nn's that will make up resnet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,),
                nn.BatchNorm2d(self.expansion * planes),)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(pl.LightningModule):
    """Lightning module used to set up the trainable module (resnet)"""
    def __init__(self, block, num_blocks, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.classes = num_classes
        self.in_planes = 64
        self.lr = lr

        #set up first layer with no downsizing
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        #create layers depending on resnet
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Making layers for resnet18"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """The layout of each training step"""
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
        """Optimizer including learning rate scheduler"""
        #sets up optimizer
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,)
        
        #seting up learning rate scheduler
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=0.1,
                epochs=self.trainer.max_epochs,
                total_steps=self.trainer.estimated_stepping_batches),
                "interval": "step",}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        out = self(x)
        loss = F.cross_entropy(out,y)
        self.log('train_loss', loss,on_step=True,on_epoch=True)
        return loss 
    
    def evaluate(self, batch, stage=None):
        """Basic function used to retrive accuracy and loss for test and validation"""
        x,y = batch
        out = self(x)
        loss = F.cross_entropy(out,y)
        out = nn.Softmax(-1)(out) 
        logits = torch.argmax(out,dim=1)
        acc = self.accuracy(logits, y)  

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
    """Lighting module to set up batches and load data into module
        downloads data and checks if already installed
        splits and batchises the data"""
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 0

        CIFAR10_mean = [0.4914, 0.4822, 0.4465]
        CIFAR10_std = [0.2470, 0.2435, 0.2616]
        #set up transform for train split
        self.train_transform = torchvision.transforms.Compose([ 
                                RandomCrop(32, padding=4),
                                RandomHorizontalFlip(),
                                ToTensor(),
                                Normalize(CIFAR10_mean, CIFAR10_std)])
        
        #set up transform for test split
        self.test_transform = Compose([ToTensor(), Normalize(CIFAR10_mean, CIFAR10_std)])

        self.train = torchvision.datasets.CIFAR10(
            root='./CIFAR10_data', 
            download=True,
            train=True, 
            transform=self.train_transform)
        
        self.test = torchvision.datasets.CIFAR10(
            root='./CIFAR10_data', download=True, 
            train=False, 
            transform=self.test_transform)
        
        print('Data loaded')

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=self.num_workers)    

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.test, self.batch_size, num_workers=self.num_workers)
    
def main():
    #set up variables
    max_epochs = 30

    #set batch size
    BATCH_SIZE = 256

    #loading data
    data = load_CIFAR10data(BATCH_SIZE)

    #creating model
    mod = ResNet18()

    #setup trainer with logging of results
    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        accelerator='gpu', 
        devices=1,
        logger=CSVLogger(save_dir="logs/"), #save parameters into csv
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)])
    
    #train then test module
    trainer.fit(mod, data)
    trainer.test(mod, data)

    #get metrics for plotting lr, loss, acc
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    metrics.dropna(axis=1, how="all").head()
    plot = sn.relplot(data=metrics, kind="line")
    plot.savefig(f"{trainer.logger.log_dir}/graph.png")


if __name__ == '__main__': main()