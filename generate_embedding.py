import os
import subprocess
import shutil
import pandas as pd
import sys

from extract_features import extract_features

MODEL_WINDOW_SIZE = 64

def repeat_last_row(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.iloc[-1:,:]], axis=0)

if __name__ == "__main__":
    MODEL_NAME = sys.argv[1]

    dfs = []
    sequence_id_to_file = {}
    for root, dirs, files in os.walk("data"):
        if dirs:
            raise ValueError(f"Subdirectories are not allowed in the data/ folder. Found: {dirs}")
        for sequence_id, file in enumerate(sorted(list(files))):
            if file.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(root, file))
                target_address = file.replace(".parquet", "")
                dfs.append(repeat_last_row(extract_features(df, target_address,  sequence_id)))
                sequence_id_to_file[sequence_id] = target_address

    df = pd.concat(dfs, axis = 0)
    
    os.makedirs("temp-data")
    df.to_parquet("temp-data/feature-data.parquet")

    subprocess.run(["sequifier", "preprocess"], check=True)

    subprocess.run(["mv", "data/feature-data-split0.parquet", "temp-data/feature-data-split0.parquet"])

    if not os.path.exists(f"models/{MODEL_NAME.lower()}.onnx"):
        raise Exception(f"Error: model file 'models/{MODEL_NAME.lower()}.onnx' does not exist.")

    subprocess.run(["sequifier", "infer", "--model-path", f"models/{MODEL_NAME.lower()}.onnx"], check=True)

    embeddings = pd.read_parquet(f"outputs/embeddings/{MODEL_NAME.lower()}-embeddings.parquet")

    shutil.rmtree("temp-data")
    shutil.rmtree("outputs")
    os.remove("configs/metadata_configs/feature-data.json")

    os.makedirs("embeddings", exist_ok=True)

    for sequence_id, sequence_group in embeddings.groupby("sequenceId"):
        file_path = f"embeddings/{sequence_id_to_file[sequence_id]}.parquet"
        sequence_group.to_parquet(file_path)