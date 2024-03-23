from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transformers import BartForConditionalGeneration, BartConfig, PretrainedConfig
from dataset_modules.bart_dataset import BartDataset
from pipelines.pipeline import BartPipeline

from models.bart_model import BartTranslator
from manual_tokenizer.load_tokenizer import load_tokenizer


def main():
    # | Load Tokenizer |
    tokenizer_path = "/home/kangnam/project/practice/bart_translator/manual_tokenizer/ko_en_tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    # | Load Train Dataset |
    train_data_path = (
        "/home/kangnam/datasets/raw/ko_en_translation/training/raw/ko_en_train_set.json"
    )
    train_dataset = BartDataset(train_data_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=8)

    # | Load Valid Dataset |
    valid_data_path = "/home/kangnam/datasets/raw/ko_en_translation/validation/raw/ko_en_valid_set.json"
    valid_dataset = BartDataset(valid_data_path, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=16, num_workers=8)

    # | Load config of bart-base |
    config = PretrainedConfig.from_pretrained("facebook/bart-base")
    config.vocab_size = 50000
    config.bos_token_id = 0
    config.eos_token_id = 1
    config.pad_token_id = 2
    config.decoder_start_token_id = 1
    config.forced_eos_token_id = 1

    # | Model Load |
    model = BartForConditionalGeneration(config)
    logger = WandbLogger(project="bart_translator")
    pipeline = BartPipeline(model=model, lr=1e-4, tokenizer=tokenizer)

    # | Train |
    trainer = pl.Trainer(logger=logger, precision="16-mixed")
    trainer.fit(pipeline, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
