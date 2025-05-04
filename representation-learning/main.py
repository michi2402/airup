import yaml
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import util.data_generator as data_generator
from sklearn.model_selection import train_test_split
from util.QADataset import QADataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config('config.yaml')
    full_dataset = data_generator.DataGenerator(config['train_data_path'], config['data_set_output_path'],
                                                generate_new_dataset=True)

    logger.info("Setting up for training...")
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    model = BertForSequenceClassification.from_pretrained(config['model_name'], num_labels=2)

    train_data, val_data = train_test_split(full_dataset.data, test_size=config['validation_split'],
                                            random_state=config['seed'])
    train_dataset = QADataset(train_data, tokenizer)
    val_dataset = QADataset(val_data, tokenizer)

    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./results"),
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=float(2e-5),
        weight_decay=.01,
        load_best_model_at_end=True,
        logging_dir=config.get("logging_dir", "./logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    main()
