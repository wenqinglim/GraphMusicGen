import re
import os
import time
import sys
import multiprocessing
import itertools
import argparse
from itertools import product

import numpy as np
from tqdm import tqdm

import constants
from constants import IdToken, LengthToken


def split_string(s):
    # This regex pattern matches a letter followed by one or more digits
    pattern = re.compile(r'[a-zA-Z]\d+')
    # Find all matches in the string
    matches = pattern.findall(s)
    return matches


def is_melodic(phrase):
    return phrase[0].isupper()


def process_phrases(phrases):
    # Create structure matrix: 2 (Non-melodic and Melodic) X MAX_STRUCTURE_LEN 
    s_tensor = np.zeros((2, constants.MAX_STRUCTURE_LEN), np.int16)
    # Create content matrix: 2 (Non-melodic and Melodic) x MAX_STRUCTURE_LEN x 3 x 2
    c_tensor = np.zeros((2, constants.MAX_STRUCTURE_LEN, 3, 2), np.int16)
    
    c_tensor[:, :, :, 0] = IdToken.PAD.value
    c_tensor[:, :, 0, 0] = IdToken.SOS.value
    c_tensor[:, :, :, 1] = LengthToken.PAD.value
    c_tensor[:, :, 0, 1] = LengthToken.SOS.value
    
    
    added_phrases = {}
        
    for idx, phrase in enumerate(phrases):
        if idx >= constants.MAX_STRUCTURE_LEN:
            break
            
        melodic_idx = int(is_melodic(phrase)) # melodic: 1; non-melodic: 0
        empty_idx = int(~is_melodic(phrase))
            
        # Add structure activation
        # print(s_tensor)
        s_tensor[melodic_idx, idx] = 1
        
        # Add content repetition
        if phrase in added_phrases.keys():
            phrase_id = added_phrases[phrase]
        else:
            # Assign new phrase_id to phrase
            phrase_id = len(added_phrases)
            added_phrases[phrase] = phrase_id
            
        phrase_id = min(phrase_id, constants.MAX_ID_TOKEN)
        c_tensor[melodic_idx, idx, 1, 0] = phrase_id

        # Add content n_bars
        n_bars = min(int(phrase[1:]), constants.MAX_LEN_TOKEN)
        c_tensor[melodic_idx, idx, 1, 1] = n_bars
        

        c_tensor[melodic_idx, idx, 2, 0] = IdToken.EOS.value
        c_tensor[melodic_idx, idx, 2, 1] = LengthToken.EOS.value
        c_tensor[empty_idx, idx, 1, 0] = IdToken.EOS.value
        c_tensor[empty_idx, idx, 1, 1] = LengthToken.EOS.value
        
    return c_tensor, s_tensor 
        


def preprocess_structure_file(song_idx, structure_dir, dest_dir, n_phrases):
    saved_samples = 0 # Restart count for every song
    label_paths = ["human_label1"]

    for label_path in label_paths:
        # try:

            f = open(f"{structure_dir}/{song_idx}/{label_path}.txt", "r")
            structure = f.read()
            phrases = split_string(structure)
            # print(f"Structure: {structure}")
            # print(f"Phrase length: {len(phrases)}")
            
            # phrases = split_phrase_string_to_max(phrases)

            # phrase_songs = split_song_into_phrases(pproll_song, phrases, resolution)
            # print(f"Phrase songs length: {len(phrase_songs)}")
            
            c_tensor, s_tensor = process_phrases(phrases)
            
            # Save sample (content and structure) to file
            # filename = os.path.basename(midi_filepath)

            sample_filepath = os.path.join(
                dest_dir, song_idx+str(saved_samples))
            np.savez(sample_filepath, c_tensor=c_tensor, s_tensor=s_tensor)
            print(f"saved to {sample_filepath}")

            saved_samples += 1
            
                
        # except Exception as e:
        #     print(f"Skipped {midi_filepath}: Error: {e}")
        #     continue

    
    
def preprocess_structure_dataset(structure_dir, preprocessed_dir, n_phrases, n_files=None, n_workers=1):
    

    print("Starting preprocessing")
    start = time.time()

    # Visit recursively the directories inside the dataset directory
    with multiprocessing.Pool(n_workers) as pool:

        # song_idxs = [f for f in os.listdir(structure_dir) if not f.startswith('.')]
        song_idxs = [f for f in os.listdir(structure_dir) if 
                     ((not f.startswith('.')) and os.path.isdir(os.path.join(structure_dir, f)))
                    ]



        fn_gen = itertools.chain(
            (song_idx, structure_dir, preprocessed_dir, n_phrases)
                for song_idx in song_idxs
        )

        r = list(tqdm(pool.starmap(preprocess_structure_file, fn_gen),
                           total=n_files))

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Preprocessing completed in (h:m:s): {:0>2}:{:0>2}:{:05.2f}"
          .format(int(hours), int(minutes), seconds))
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocesses a MIDI dataset. MIDI files can be arranged " 
            "hierarchically in subdirectories, similarly to the Lakh MIDI "
            "Dataset (lmd_matched) and the MetaMIDI Dataset."
    )
    parser.add_argument(
        'structure_dir',
        type=str, 
        help='Directory of the structure labels dataset.'
    )
    parser.add_argument(
        'preprocessed_dir',
        type=str,
        help='Directory to save the preprocessed dataset.'
    )
    parser.add_argument(
        '--n_phrases',
        type=int,
        default=12,
        help="Number of phrases for each sequence of the resulting preprocessed "
            "dataset. Defaults to 12 bars."
    )
    parser.add_argument(
        '--n_files',
        type=int,
        help="Number of files in the MIDI dataset. If set, the script "
            "will provide statistics on the time remaining."
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=1,
        help="Number of parallel workers. Defaults to 1."
    )

    args = parser.parse_args()
    
    # Create the output directory if it does not exist
    if not os.path.exists(args.preprocessed_dir):
        os.makedirs(args.preprocessed_dir)

    preprocess_structure_dataset(args.structure_dir, args.preprocessed_dir, 
                            args.n_phrases, args.n_files,
                            n_workers=args.n_workers)
