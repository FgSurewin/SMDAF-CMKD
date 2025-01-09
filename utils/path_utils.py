import os
from datetime import datetime
from pathlib import Path
from glob import glob
import shutil
import json


class PathUtils:
    folder_name = "test_brandon"  # raw_data
    root_path = Path(os.path.abspath("."), "data")
    raw_data_path = Path(root_path / folder_name)

    @staticmethod
    def create_dir(folder_path, show_message=True):
        if not os.path.exists(folder_path):
            if show_message:
                print("Creating folder: {}".format(folder_path))
            os.makedirs(folder_path)

    @staticmethod
    def remove_dir(folder_path):
        if os.path.exists(folder_path):
            print("Removing folder: {}".format(folder_path))
            shutil.rmtree(folder_path)

    @staticmethod
    def generate_output_folder_name(clip_duration=None, step_size=None):
        today = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        step = f"${step_size}" if step_size is not None else ""
        folder_name = (
            f"{today}@{str(clip_duration)}{step}"
            if clip_duration is not None
            else today
        )
        return folder_name

    @staticmethod
    def generate_cache_folder_name(cache_name):
        today = datetime.now().strftime("%H_%M")
        folder_name = f"{today}_{cache_name}.csv"
        return folder_name

    @staticmethod
    def save_json(json_data, json_path):
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

    @staticmethod
    def read_json(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
        return json_data

    @staticmethod
    def search_files(file_path, pattern="*"):
        return glob(os.path.join(file_path, pattern))


if __name__ == "__main__":
    # root_path = Path(os.path.abspath("."), "data")
    # PathUtils.create_dir(Path(root_path) / "test" / "inside")
    # print(PathUtils.root_path)
    print("Done")
