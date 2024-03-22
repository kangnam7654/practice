from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from models.bert_classifier import BertClassifier
from pipelines.bert_pipeline import BertPipeline
from dataset_modules.bert_dataset import BertDataset

def main():
    train_dataset = BertDataset(train=True)
    valid_dataset = BertDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=8)
    valid_loader = DataLoader(valid_dataset, batch_size=8)
    
    logger = WandbLogger(project="Test")

    model = BertClassifier()
    pipeline = BertPipeline(model=model, lr=2e-5)

    trainer = pl.Trainer(logger=logger)
    trainer.fit(model = pipeline, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == "__main__":
    main()