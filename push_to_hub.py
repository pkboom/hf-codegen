from huggingface_hub import snapshot_download
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import glob
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

REPO_ID = "pkboom/codegen"
FEATHER_FORMAT = "ftr"

if __name__ == "__main__":
    folder_path = snapshot_download(
        repo_id=REPO_ID, allow_patterns=f"*.{FEATHER_FORMAT}", repo_type="dataset"
    )
    print(folder_path)

    feather_files = glob.glob(f"{folder_path}/*.{FEATHER_FORMAT}")
    print(len(feather_files))

    all_dfs = []

    for feather_file in tqdm(feather_files):
        df = pd.read_feather(feather_file)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs)
    print(f"Final DF prepared containing {len(final_df)} rows.")

    dataset = Dataset.from_pandas(final_df)
    dataset.push_to_hub(
        "codegen-v2",
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )
