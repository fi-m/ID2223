class Helper:
    def __init__(self):
        pass

    def save_to_disk(self, dataset):
        """
        Save the dataset to disk

        params:
            dataset: the dataset to save: dict{"train": train_dataset, "test": test_dataset}
        """

        print("SAVING DATASET TO DISK")
        __import__("pprint").pprint(dataset["train"][0])

        dataset["train"].save_to_disk("train")
        dataset["test"].save_to_disk("test")

        print("DATASET SAVED TO DISK")
        __import__("pprint").pprint(dataset["train"][0])

    def _remove_from_disk(self):
        import os

        os.system("rm -rf train")
        os.system("rm -rf test")

    def upload_to_drive(self, LOCAL=False):
        import os.path
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.errors import HttpError

        # Parse the environment variable to a json file
        try:
            with open("./credentials.json", "w") as f:
                f.write(os.environ["GOOGLE_CLIENT_SECRET"])
        except:
            print("No credentials found in environment variable")

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

        self._remove_from_disk()

    def download_from_drive(self, LOCAL=False):
        import io
        import os.path
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.errors import HttpError

        # Parse the environment variable to a json file
        try:
            with open("./credentials.json", "w") as f:
                f.write(os.environ["GOOGLE_CLIENT_SECRET"])
        except:
            print("No credentials found in environment variable")

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

            print("DOWNLOADING TRAIN DATASET")
            results = (
                service.files()
                .list(
                    q=f"'{os.environ['TRAIN_FOLDER']}' in parents",
                    pageSize=10,
                    fields="nextPageToken, files(id, name)",
                )
                .execute()
            )
            items = results.get("files", [])
            if not items:
                print("No files found.")
            else:
                for item in items:
                    os.makedirs("train", exist_ok=True)
                    print("DOWNLOADING", item["name"])
                    request = service.files().get_media(fileId=item["id"])
                    fh = io.FileIO(os.path.join("train", item["name"]), "wb")
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print("Download %d%%." % int(status.progress() * 100))

            print("DOWNLOADING TEST DATASET")
            results = (
                service.files()
                .list(
                    q=f"'{os.environ['TEST_FOLDER']}' in parents",
                    pageSize=10,
                    fields="nextPageToken, files(id, name)",
                )
                .execute()
            )
            items = results.get("files", [])
            if not items:
                print("No files found.")
            else:
                for item in items:
                    os.makedirs("test", exist_ok=True)
                    print("DOWNLOADING", item["name"])
                    request = service.files().get_media(fileId=item["id"])
                    fh = io.FileIO(os.path.join("test", item["name"]), "wb")
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print("Download %d%%." % int(status.progress() * 100))

        except HttpError as e:
            print(e)

    def upload_model_to_drive(self, LOCAL=False, PATH_TO_MODEL=""):
        import os.path
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.errors import HttpError

        # Parse the environment variable to a json file
        try:
            with open("./credentials.json", "w") as f:
                f.write(os.environ["GOOGLE_CLIENT_SECRET"])
        except:
            print("No credentials found in environment variable")

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
            for file in os.listdir("PATH_TO_MODEL"):
                print("UPLOADING", file)
                file_metadata = {"name": file, "parents": [os.environ["MODEL_FOLDER"]]}
                media = MediaFileUpload(os.path.join("train", file), resumable=True)
                service.files().create(
                    body=file_metadata, media_body=media, fields="id"
                ).execute()

        except HttpError as e:
            print(e)
