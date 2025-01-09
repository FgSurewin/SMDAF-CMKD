import os
import shutil
import argparse
from google.cloud import storage
from google.oauth2 import service_account


def zip_folder(folder_path):
    shutil.make_archive(folder_path, "zip", folder_path)


def upload_blob(bucket_name, source_file_name, destination_blob_name, creds):
    credentials = service_account.Credentials.from_service_account_file(creds)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    try:
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")

        # Delete the local file after successful upload
        os.remove(source_file_name)
        print(f"File {source_file_name} has been deleted locally.")
    except Exception as e:
        print(f"Failed to upload or delete file: {e}")


def main(args):
    folder_path = args.folder_path
    zip_folder(folder_path)
    zip_file_path = folder_path + ".zip"
    upload_blob(
        bucket_name=args.bucket_name,
        source_file_name=zip_file_path,
        destination_blob_name=zip_file_path.split("/")[-1],
        creds=args.cred_path,
    )


if __name__ == "__main__":
    print("------------Upload Checkpoint to Cloud----------------")
    print("--------------------Zipping Folder--------------------")
    parser = argparse.ArgumentParser(
        description="Upload checkpoints to gcp server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ------------------------------- Dataset Args ------------------------------- #
    parser.add_argument("--folder_path", type=str, required=True, help="folder path")
    parser.add_argument(
        "--bucket_name",
        type=str,
        default="global-wacv-2025",
        help="folder path",
    )
    parser.add_argument(
        "--cred_path",
        type=str,
        default="./credentials/nsf-2131186-18936-e9872861c262.json",
        help="folder path",
    )
    # ------------------------------------ XX ------------------------------------ #
    args = parser.parse_args()

    # ----------------------------------- Main ----------------------------------- #
    main(args)
