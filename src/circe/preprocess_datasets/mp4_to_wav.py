import argparse
import os
import glob
import subprocess
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

def parse_args(args=None):
    parser = argparse.ArgumentParser("Convert .mp4 from YT to .wav")
    parser.add_argument("-i", "--in_folder", type=str, required=True,
        help="Input folder with .mp4 to convert to .wav")
    parser.add_argument("-o", "--out_folder", type=str, required=True,
        help="Output folder to store .wav files")
    parser.add_argument("-p", "--num_processes", type=int, default=8,
        help="No. processes to use. (Default: 8)")
    return parser.parse_args(args=args)

def command(infile: str, outfile: str) -> str:
    return f"ffmpeg -i \"{infile}\" -ar 44100 \"{outfile}\""

def convert_audio(infile: str, out_folder: str) -> None:
    outfile = os.path.join(out_folder, f"{os.path.splitext(os.path.basename(infile))[0]}.wav")
    command_str = command(infile, outfile)
    subprocess.call(command_str, shell=True)

def main(args):
    audios = glob.glob(os.path.join(args.in_folder, "*.mp4"))
    os.makedirs(args.out_folder, exist_ok=True)

    with Pool(processes=args.num_processes) as p:
        with tqdm(total=len(audios)) as pbar:
            for _ in p.imap_unordered(
                partial(convert_audio, out_folder=args.out_folder), audios
            ):
                pbar.update()

if __name__ == "__main__":
    main(parse_args())
