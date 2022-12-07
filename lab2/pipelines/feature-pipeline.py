#################       IMPORTS       #################
from helper import Helper

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
            "google-api-python-client",
            "google-auth-httplib2",
            "google-auth-oauthlib",
        ]
    )
    .apt_install(["ffmpeg"])
)


################     Initate pipeline    ################
# @stub.function(image=image, secret=modal.Secret.from_name("lab2_whisper"), timeout=1000)
def main():
    # Log into huggingface
    import os
    from huggingface_hub import login

    login(token=os.environ["HUGGINGFACE_TOKEN"])

    # Load the datasets
    from datasets import load_dataset, DatasetDict

    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "sv-SE",
        split="train+validation",
        use_auth_token=True,
    )
    common_voice["test"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "sv-SE",
        split="test",
        use_auth_token=True,
    )

    # Remove redundant columns
    common_voice = common_voice.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )

    # Cast audio to 16kHz, which whisprer uses
    from datasets import Audio

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    # Create feature extractor
    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    # Create tokenizer
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="Swedish", task="transcribe"
    )

    # Helper function to tokenize and extract features
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch


    print("Before preprocessing")
    __import__('pprint').pprint(common_voice)
    print("ROW= ", common_voice["train"][0])

    # Prepare the datasets
    common_voice = common_voice.map(
        prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
    )

    print("After preprocessing")
    __import__('pprint').pprint(common_voice)
    print("ROW= ", common_voice["train"][0])

    helper = Helper()
    helper.save_to_disk(dataset=common_voice)

    # Upload the dataset to drive
    helper.upload_to_drive(LOCAL=False)


if __name__ == "__main__":
    # with stub.run():
    main()
