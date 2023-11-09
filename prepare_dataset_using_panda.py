import os
import pandas as pd
from nbformat import reads, NO_CONVERT
from tqdm import tqdm
from datasets import Dataset
from typing import Dict
from huggingface_hub import create_repo
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

MIRROR_DIRECTORY = "hf_public_repos"
DATASET_ID = "codegen"
FEATHER_FORMAT = "ftr"

# Block the following formats.
IMAGE = ["png", "jpg", "jpeg", "gif"]
VIDEO = ["mp4", "jfif"]
DOC = [
    "key",
    "PDF",
    "pdf",
    "docx",
    "xlsx",
    "pptx",
]
AUDIO = ["flac", "ogg", "mid", "webm", "wav", "mp3"]
ARCHIVE = ["jar", "aar", "gz", "zip", "bz2"]
MODEL = ["onnx", "pickle", "model", "neuron"]
OTHERS = [
    "npy",
    "index",
    "inv",
    "index",
    "DS_Store",
    "rdb",
    "pack",
    "idx",
    "glb",
    "gltf",
    "len",
    "otf",
    "unitypackage",
    "ttf",
    "xz",
    "pcm",
    "opus",
]
ANTI_FOMATS = tuple(IMAGE + VIDEO + DOC + AUDIO + ARCHIVE + OTHERS)


def create_repo_on_hf(repo_id: str):
    repo_id = create_repo(
        repo_id=repo_id,
        exist_ok=True,
        repo_type="dataset",
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    ).repo_id


def filter_code_cell(cell) -> bool:
    """Filters a code cell w.r.t shell commands, etc."""
    only_shell = cell["source"].startswith("!")
    only_magic = "%%capture" in cell["source"]
    if only_shell or only_magic:
        return False
    else:
        return True


def process_file(directory_name: str, file_path: str) -> Dict[str, str]:
    """Processes a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            if file_path.endswith("ipynb"):
                # Code courtesy: Chansung Park and Sayak Paul.
                code_cell_str = ""
                notebook = reads(content, NO_CONVERT)

                code_cells = [
                    c
                    for c in notebook["cells"]
                    if c["cell_type"] == "code"
                    if filter_code_cell(c)
                ]

                for cell in code_cells:
                    code_cell_str += cell["source"]
                content = code_cell_str
    except Exception:
        content = ""

    return {
        "repo_id": directory_name,
        "file_path": file_path,
        "content": content,
    }


def read_repository_files(directory) -> pd.DataFrame:
    """Reads the files from the locally cloned repositories."""
    file_paths = []
    df = pd.DataFrame(columns=["repo_id", "file_path", "content"])

    # Recursively find all files within the directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(ANTI_FOMATS) and all(
                k not in file_path for k in [".git", "__pycache__", "xcodeproj"]
            ):
                file_paths.append((os.path.dirname(root), file_path))
                # e.g. ('hf_public_repos/experiment/resources/js', 'hf_public_repos/experiment/resources/js/Components/InputLabel.vue')

    # Process files sequentially.
    print(f"Total file paths: {len(file_paths)}.")
    print("Reading file contents...")

    for directory_name, file_path in tqdm(file_paths):
        file_content = process_file(directory_name, file_path)

        if file_content["content"] != "":
            temp_df = pd.DataFrame.from_dict([file_content])
            df = pd.concat([df, temp_df])

    return df


if __name__ == "__main__":
    df = read_repository_files(MIRROR_DIRECTORY)

    print("DataFrame created, creating dataset...")
    create_repo_on_hf(repo_id=DATASET_ID)

    print("Serializing dataframe...")
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(
        DATASET_ID,
        private=True,
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )
