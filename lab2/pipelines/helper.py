class Helper:
    def __init__(self):
        pass

    def save_to_disk(self, dataset):
        """
        Save the dataset to disk

        params:
            dataset: the dataset to save: dict{"train": train_dataset, "test": test_dataset}
        """

        dataset["train"].save_to_disk("train")
        dataset["test"].save_to_disk("test")

    def upload_to_drive(self, LOCAL=False):
        import os.path
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.errors import HttpError

        # Parse the environment variable to a json file
        with open("./credentials.json", "w") as f:
            f.write(os.environ["GOOGLE_CLIENT_SECRET"])

        # Add scope
        scope = ["https://www.googleapis.com/auth/drive"]

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", scope)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", scope
                )
                if LOCAL:
                    flow.run_local_server(port=0)
                else:
                    flow.run_console()

                creds = flow.credentials
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        try:
            service = build("drive", "v3", credentials=creds)

            print("UPLOADING TRAIN DATASET")
            for file in os.listdir("train"):
                print("UPLOADING", file)
                file_metadata = {"name": file, "parents": [os.environ["TRAIN_FOLDER"]]}
                media = MediaFileUpload(os.path.join("train", file), resumable=True)
                service.files().create(
                    body=file_metadata, media_body=media, fields="id"
                ).execute()

            print("UPLOADING TEST DATASET")
            for file in os.listdir("test"):
                print("UPLOADING", file)
                file_metadata = {"name": file, "parents": [os.environ["TEST_FOLDER"]]}
                media = MediaFileUpload(os.path.join("test", file), resumable=True)
                service.files().create(
                    body=file_metadata, media_body=media, fields="id"
                ).execute()

        except HttpError as e:
            print(e)

        os.system("rm -rf train")
        os.system("rm -rf test")
