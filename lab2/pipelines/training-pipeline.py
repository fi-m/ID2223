#################       IMPORTS       #################
from helper import Helper
from datacollator import DataCollatorSpeechSeq2SeqWithPadding

################     Initate modal    #################
import modal

stub = modal.Stub("lab2_feature_pipeline_whisper")
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "datasets",
            "librosa",
            "jiwer",
            "huggingface_hub",
            "transformers",
            "torchaudio",
            "touch",
            "google-api-python-client",
            "google-auth-httplib2",
            "google-auth-oauthlib",
            "evaluate",
        ]
    )
    .apt_install(["ffmpeg"])
)


################     Initate pipeline    ################
@stub.function(
    image=image,
    secret=modal.Secret.from_name("lab2_whisper"),
    timeout=86300,
    gpu=True,
)
def main():
    import os

    # Check if dataset is already downloaded
    if not os.path.exists("train") and not os.path.exists("test"):
        Helper().download_from_drive(LOCAL=False)

    # Load the datasets
    from datasets import load_from_disk

    common_voice = {}
    common_voice["train"] = load_from_disk("train")
    common_voice["test"] = load_from_disk("test")

    # Load whisper
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Swedish", task="transcribe"
    )

    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="Swedish", task="transcribe"
    )

    # Create the data collator with whipser processor
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    import evaluate

    metric = evaluate.load("wer")

    # Metric function
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # convert to %WER
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # Load the model
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Create the trainer
    from transformers import Seq2SeqTrainingArguments

    training_args = Seq2SeqTrainingArguments(
        # change to a repo name of your choice
        output_dir="whisper-small-sv-SE",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=False,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )

    from transformers import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    kwargs = {
        "dataset_tags": "mozilla-foundation/common_voice_11_0",
        "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
        "dataset_args": "config: sv-SE, split: test",
        "language": "sv-SE",
        "model_name": "Whisper Small sv-SE - Lab 2",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-small",
        "tasks": "automatic-speech-recognition",
        "tags": "i-dont-know-what-im-doing",
    }

    trainer.push_to_hub(**kwargs)

    # Back up the model to drive if huggingface is down
    Helper().upload_model_to_drive(LOCAL=False, PATH_TO_MODEL="whisper-small-sv-SE")


if __name__ == "__main__":
    with stub.run():
        main()
