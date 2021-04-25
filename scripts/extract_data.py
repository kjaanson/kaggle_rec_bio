

import os
import argparse
import zipfile
import time
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, dest='data_file', help='input zip file')
    parser.add_argument('--output-path', type=str, dest='output_path', help='output path', default=".")
    args = parser.parse_args()

    print(f"Extracting {args.data_file} to {args.output_path}")

    os.makedirs(args.output_path, exist_ok=True)

    archive = zipfile.ZipFile(args.data_file, 'r')

    for zi in tqdm(archive.infolist()):
        archive.extract(zi, path=args.output_path)