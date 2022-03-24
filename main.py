from dataloader import NetDataModule
from routenet import RouteNet
import pytorch_lightning as pl

if __name__ == "__main__":
    dm = NetDataModule('./dataset/nsfnetbw', 'delay')
    model = RouteNet(32, 32, 128, 4)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, dm)