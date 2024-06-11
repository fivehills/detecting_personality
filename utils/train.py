import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW, set_seed
from utils.preprocess import preprocess_data
import config

def compute_metrics(pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    labels = pred.label_ids
    preds = pred.predictions
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    return {
        'mse': mse,
        'mae': mae
    }

def train_model():
    set_seed(config.SEED)

    train_dataset, val_dataset, test_dataset, tokenizer = preprocess_data()

    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL, num_labels=5, problem_type="regression")
    model.classifier.out_proj = torch.nn.Linear(model.config.hidden_size, model.config.num_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(AdamW(model.parameters(), lr=config.LEARNING_RATE), None)
    )

    trainer.train()
    trainer.save_model("./results/personality_model")
    tokenizer.save_pretrained("./results/personality_model")

if __name__ == "__main__":
    train_model()
    print("Training completed.")

