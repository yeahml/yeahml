import json
import os
import shutil
import yaml

# helper to create dirs if they don't already exist
def maybe_create_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("{} created".format(dir_path))
    else:
        print("{} already exists".format(dir_path))


def parse_yaml_from_path(path: str) -> dict:
    # return python dict from yaml path
    try:
        with open(path, "r") as stream:
            try:
                y = yaml.load(stream, Loader=yaml.FullLoader)
                return y
            except yaml.YAMLError as exc:
                print(exc)
                return dict()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error > Exiting: the configuration file {path} was not found"
        )


def parse_json_from_path(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as data_file:
            try:
                data = json.loads(data_file.read())
                return data
            except:
                print("Error loading json to dict for file {}".format(path))
                return dict()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error > Exiting: the configuration file {path} was not found"
        )


def create_standard_dirs(root_dir: str, wipe_dirs: bool):
    # this logic is messy
    if wipe_dirs:
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        maybe_create_dir(root_dir)
    else:
        maybe_create_dir(root_dir)

    # maybe_create_dir(root_dir + "/saver")
    # `best_params/` will hold a serialized version of the best params
    # I like to keep this as a backup in case I run into issues with
    # the saver files
    maybe_create_dir(os.path.join(root_dir, "best_params"))
    # `tf_logs/` will hold the logs that will be visible in tensorboard
    maybe_create_dir(os.path.join(root_dir, "tf_logs"))

    # `yf_logs/` will hold the custom logs
    maybe_create_dir(os.path.join(root_dir, "yf_logs"))
