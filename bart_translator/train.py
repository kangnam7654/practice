from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transformers import BartForConditionalGeneration, BartConfig, PretrainedConfig
from dataset_modules.bart_dataset import BartDataset
from pipelines.pipeline import BartPipeline

from models.bart_model import BartTranslator
from manual_tokenizer.load_tokenizer import load_tokenizer


def main():
    model_name = "facebook/bart-base"
    tokenizer_path = "/home/kangnam/project/practice/bart_translator/manual_tokenizer/ko_en_tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    data_path = (
        "/home/kangnam/datasets/raw/ko_en_translation/training/raw/ko_en_train_set.json"
    )
    dataset = BartDataset(data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=16, num_workers=8)
    config = PretrainedConfig.from_pretrained("facebook/bart-base")
    config.vocab_size = 50000
    config.bos_token_id=0
    config.eos_token_id=1
    config.pad_token_id=2
    config.decoder_start_token_id=1
    config.forced_eos_token_id=1

    model = BartForConditionalGeneration(config)
    logger = WandbLogger(project="Test")
    pipeline = BartPipeline(model=model, lr=1e-5, tokenizer=tokenizer)

    trainer = pl.Trainer(logger=logger, precision="16-mixed")
    trainer.fit(pipeline, train_dataloaders=loader)


if __name__ == "__main__":
    main()
