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
import pypianoroll as pproll
import muspy

import constants
from constants import PitchToken, DurationToken


def split_string(s):
    # This regex pattern matches a letter followed by one or more digits
    pattern = re.compile(r'[a-zA-Z]\d+')
    # Find all matches in the string
    matches = pattern.findall(s)
    return matches


def split_song_into_phrases(pproll_song, phrases, resolution):
    muspy_conv = muspy.from_pypianoroll(pproll_song)

    n_beats = [track.notes[-1].time//resolution for track in muspy_conv.tracks]
    num_beats_per_phrase = 4*constants.MAX_PHRASE_LEN
    num_bars = max(n_beats)//4
    max_phrase_len_res = 4*resolution*constants.MAX_PHRASE_LEN # tokens in max number of bars
    
    # print(f"Num bars: {num_bars}")

    tracks = []
    # For each piece, interate through each track
    for track in muspy_conv.tracks:
        # For each track, split track into bars
        bars = [[] for i in range(num_bars+1)]

        for note in track.notes:
            bar_num = note.time//(4*resolution)
            bars[bar_num].append(note)
        tracks.append(bars)
        
    # Split one song into phrases
    start_bar_idx = 0
    phrase_songs=[]
    for phrase in phrases: 
        num_bars = int(phrase[1:])
        end_bar_idx = start_bar_idx+ min(num_bars, max_phrase_len_res)
        phrase_song = muspy.Music(resolution=resolution)
        phrase_tracks = [muspy.Track(notes=[note for t in track[start_bar_idx:end_bar_idx] for note in t]) for track in tracks]
        phrase_song.tracks = phrase_tracks
        for idx, track_type in enumerate(["MELODY", "BRIDGE", "PIANO"]):
            phrase_song.tracks[idx].name = track_type
        phrase_songs.append(phrase_song)
        start_bar_idx += num_bars
        
    return phrase_songs


def process_track_notes(tracks_notes, resolution):
    tracks_content = []
    tracks_structure = []

    max_phrase_len_res = 4*resolution*constants.MAX_PHRASE_LEN # tokens in max number of bars

    for notes in tracks_notes:

        # track_content: length x MAX_SIMU_TOKENS x 2
        # This is used as a basis to build the final content tensors for
        # each sequence.
        # The last dimension contains pitches and durations. int16 is enough
        # to encode small to medium duration values.
        track_content = np.zeros((max_phrase_len_res, constants.MAX_SIMU_TOKENS, 2), 
                                np.int16)

        track_content[:, :, 0] = PitchToken.PAD.value
        track_content[:, 0, 0] = PitchToken.SOS.value
        track_content[:, :, 1] = DurationToken.PAD.value
        track_content[:, 0, 1] = DurationToken.SOS.value

        # Keeps track of how many notes have been stored in each timestep
        # (int8 imposes MAX_SIMU_TOKENS < 256)
        notes_counter = np.ones(max_phrase_len_res, dtype=np.int8)

        # Todo: np.put_along_axis?
        for note in notes:
            # Insert note in the lowest position available in the timestep

            t = note.time%max_phrase_len_res
            
            # if t >= max_phrase_len_res:
            #     # Skip note if it exceeds max phrase length
            #     continue

            if notes_counter[t] >= constants.MAX_SIMU_TOKENS-1:
                # Skip note if there is no more space
                continue

            pitch = max(min(note.pitch, constants.MAX_PITCH_TOKEN), 0)
            track_content[t, notes_counter[t], 0] = pitch
            dur = max(min(note.duration, constants.MAX_DUR_TOKEN + 1), 1)
            track_content[t, notes_counter[t], 1] = dur-1
            notes_counter[t] += 1
        # print(f"num notes: {notes_counter}")
        # Add EOS token
        t_range = np.arange(0, max_phrase_len_res)
        track_content[t_range, notes_counter, 0] = PitchToken.EOS.value
        track_content[t_range, notes_counter, 1] = DurationToken.EOS.value

        # Get track activations, a boolean tensor indicating whether notes
        # are being played in a timestep (sustain does not count)
        # (needed for graph rep.)
        activations = np.array(notes_counter-1, dtype=bool)

        tracks_content.append(track_content)
        tracks_structure.append(activations)
        
        # n_tracks x length x MAX_SIMU_TOKENS x 2
        c_tensor = np.stack(tracks_content, axis=0)

        # n_tracks x length
        s_tensor = np.stack(tracks_structure, axis=0)
    
    return c_tensor, s_tensor

def preprocess_midi_file(midi_dataset_dir, song_idx, structure_dir, dest_dir, n_bars, resolution):
    saved_samples = 0 # Restart count for every song
    label_paths = ["human_label1"]
    midi_filepath = f"{midi_dataset_dir}/{song_idx}/{song_idx}.mid"
    for label_path in label_paths:
        try:

            # print(midi_filepath)

            pproll_song = pproll.read(midi_filepath, resolution=resolution)
            muspy_song = muspy.read(midi_filepath)

            # Only accept songs that have a time signature of 4/4 and no time changes
            for t in muspy_song.time_signatures:
                # print(t)
                if t.numerator == 3 or t.denominator != 4:
                    print(f"Song skipped {midi_filepath} ({t.numerator}/{t.denominator} time signature)")
                    continue
                else:
                    print(f"Song accepted! {midi_filepath} with time signature {t.numerator}/{t.denominator}")


            f = open(f"{structure_dir}/{song_idx}/{label_path}.txt", "r")
            structure = f.read()
            phrases = split_string(structure)
            # print(f"Structure: {structure}")
            # print(f"Phrase length: {len(phrases)}")

            phrase_songs = split_song_into_phrases(pproll_song, phrases, resolution)
            # print(f"Phrase songs length: {len(phrase_songs)}")

            for phrase_song in phrase_songs:

                tracks_notes = [track.notes for track in phrase_song.tracks]
                c_tensor, s_tensor = process_track_notes(tracks_notes, resolution)
                # print(f"C tensor shape: {c_tensor.shape}, S tensor shape: {s_tensor.shape}")

                # Save sample (content and structure) to file
                filename = os.path.basename(midi_filepath)

                sample_filepath = os.path.join(
                    dest_dir, filename+str(saved_samples))
                np.savez(sample_filepath, c_tensor=c_tensor, s_tensor=s_tensor)
                print(f"saved to {sample_filepath}")

                saved_samples += 1
                
        except:
            print(f"Skipped {midi_filepath}")
            continue

    
    
def preprocess_midi_dataset(midi_dataset_dir, structure_dir, preprocessed_dir, n_bars, 
                            resolution, n_files=None, n_workers=1):
    

    print("Starting preprocessing")
    start = time.time()

    # Visit recursively the directories inside the dataset directory
    with multiprocessing.Pool(n_workers) as pool:

        song_idxs = os.listdir(midi_dataset_dir)

        fn_gen = itertools.chain(
            (midi_dataset_dir, song_idx, structure_dir, preprocessed_dir, n_bars, resolution)
                for song_idx in song_idxs
        )
        # print(list(fn_gen))

        r = list(tqdm(pool.starmap(preprocess_midi_file, fn_gen),
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
        'midi_dataset_dir',
        type=str, 
        help='Directory of the MIDI dataset.'
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
        '--n_bars',
        type=int,
        default=2,
        help="Number of bars for each sequence of the resulting preprocessed "
            "dataset. Defaults to 2 bars."
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=8,
        help="Number of timesteps per beat. When set to r, given that only "
            "4/4 songs are preprocessed, there will be 4*r timesteps in a bar. "
            "Defaults to 8."
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

    preprocess_midi_dataset(args.midi_dataset_dir, args.structure_dir, args.preprocessed_dir, 
                            args.n_bars, args.resolution, args.n_files,
                            n_workers=args.n_workers)
