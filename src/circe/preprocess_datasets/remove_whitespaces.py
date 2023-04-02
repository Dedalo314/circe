import argparse
import os
import glob
import subprocess
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

def parse_args(args=None):
    parser = argparse.ArgumentParser("Remove whitespaces from filenames, done inplace")
    parser.add_argument("-i", "--in_folder", type=str, required=True,
        help="Input folder with .wav to extract quantized features with EnCodec")
    parser.add_argument("-p", "--num_processes", type=int, default=8,
        help="No. processes to use. (Default: 8)")
    return parser.parse_args(args=args)

def rename_file(infile: str) -> None:
    new_name = os.path.basename(infile).replace(" ", "_")
    new_filename = os.path.join(os.path.dirname(infile), new_name)
    os.rename(infile, new_filename)

def main(args):
    audios = glob.glob(os.path.join(args.in_folder, "*.wav"))

    with Pool(processes=args.num_processes) as p:
        with tqdm(total=len(audios)) as pbar:
            for _ in p.imap_unordered(rename_file, audios):
                pbar.update()

if __name__ == "__main__":
    main(parse_args())
