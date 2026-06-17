import os
import argparse
import shutil
from datetime import datetime
import pandas as pd

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None


DEFAULT_DATASET = "martj42/international-football-results"
DEFAULT_FILENAME = "results.csv"


def download_kaggle_dataset(dataset=DEFAULT_DATASET, download_dir="kaggle_download"):
    if KaggleApi is None:
        raise RuntimeError("kaggle package not available or KaggleApi import failed. Install kaggle and configure your API token.")

    api = KaggleApi()
    api.authenticate()
    os.makedirs(download_dir, exist_ok=True)
    api.dataset_download_files(dataset, path=download_dir, unzip=True)
    downloaded = os.path.join(download_dir, DEFAULT_FILENAME)
    if not os.path.exists(downloaded):
        # try searching for any csv inside download_dir
        for fname in os.listdir(download_dir):
            if fname.endswith('.csv'):
                return os.path.join(download_dir, fname)
        raise FileNotFoundError(f"Expected {DEFAULT_FILENAME} in {download_dir}")
    return downloaded


def merge_results(local_path="results.csv", downloaded_path=None, backup=True):
    if downloaded_path is None:
        raise ValueError("downloaded_path must be provided")

    if not os.path.exists(local_path):
        # If no local file, just copy
        shutil.copy(downloaded_path, local_path)
        print(f"Copied {downloaded_path} to {local_path}")
        return local_path

    df_local = pd.read_csv(local_path)
    df_new = pd.read_csv(downloaded_path)

    # Ensure date parsing
    df_local['date'] = pd.to_datetime(df_local['date'], errors='coerce')
    df_new['date'] = pd.to_datetime(df_new['date'], errors='coerce')

    combined = pd.concat([df_local, df_new], ignore_index=True)

    # Deduplicate by a conservative key: date, home_team, away_team, home_score, away_score
    combined = combined.drop_duplicates(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])

    combined = combined.sort_values('date').reset_index(drop=True)

    if backup:
        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        backup_path = f"{local_path}.bak.{ts}"
        shutil.copy(local_path, backup_path)
        print(f"Backup saved to {backup_path}")

    combined.to_csv(local_path, index=False)
    print(f"Merged dataset saved to {local_path} (rows: {len(combined)})")
    return local_path


def main():
    parser = argparse.ArgumentParser(description="Download and merge latest FIFA results from Kaggle")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset identifier (owner/dataset)")
    parser.add_argument("--local", default=DEFAULT_FILENAME, help="Local results.csv path")
    parser.add_argument("--download-dir", default="kaggle_download", help="Temporary download directory")
    parser.add_argument("--no-backup", dest='backup', action='store_false', help="Disable creating a backup of the existing local file")
    args = parser.parse_args()

    print("Starting Kaggle dataset download and merge")
    downloaded = download_kaggle_dataset(args.dataset, download_dir=args.download_dir)
    merge_results(local_path=args.local, downloaded_path=downloaded, backup=args.backup)


if __name__ == "__main__":
    main()
