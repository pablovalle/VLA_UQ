from huggingface_hub import snapshot_download
import argparse
import os

def download_model(repo_id):
    current_path = os.getcwd()

    # Extract model name after last slash (e.g., "someuser/model-name" → "model-name")
    model_name = repo_id.split("/")[-1]
    target_path = os.path.join(current_path, model_name)

    print(f"Downloading model '{repo_id}' into folder '{model_name}'...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_path,
        local_dir_use_symlinks=False
    )

    print("✅ Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face model into a folder named after the model.")
    parser.add_argument("repo_id", type=str, help="Model repo ID (e.g. 'bert-base-uncased' or 'user/model-name')")
    
    args = parser.parse_args()
    download_model(args.repo_id)
