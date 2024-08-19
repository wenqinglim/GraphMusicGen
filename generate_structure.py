import argparse
import os
import json
import time
import math

import torch
import os
from matplotlib import pyplot as plt

import generation_config
import constants
from constants import IdToken, LengthToken

from model_structure import VAE
from utils import set_seed
from utils import mtp_from_logits, muspy_from_mtp, set_seed
from utils import print_divider
# from utils import loop_muspy_music, save_midi, save_audio
# from plots import plot_pianoroll, plot_structure


# TODO: Figure out if s_cond affects s_logits if it is partially filled.
def generate_structure(vae, z, s_cond=None, s_tensor_cond=None):
    # Decoder pass to get structure and content logits
    s_logits, c_logits = vae.decoder(z, s_cond)
    print(f"s_logits from vae decoder: {s_logits.shape}")
    print(f"c_logits from vae decoder: {c_logits.shape}")

    s_tensor = (
        s_tensor_cond
        if s_tensor_cond is not None
        else vae.decoder._binary_from_logits(s_logits)
    )

    # Build (n_batches x n_bars x n_tracks x n_timesteps x Sigma x d_token)
    # multitrack pianoroll tensor containing logits for each activation and
    # hard silences elsewhere
    # mtp = mtp_from_logits(c_logits, s_tensor)

    # Build structure from content and structure tensors
    # E.g. i3-A4-B4-A4-o2 is represented by:
    # [{phrase_id: 0, len: 3, is_melody: False},
    # {phrase_id: 1, len: 4, is_melody: True},
    # {phrase_id: 2, len: 4, is_melody: True},
    # {phrase_id: 1, len: 4, is_melody: True},
    # {phrase_id: 12, len: 2, is_melody: False}]

    print(f"num activated nodes: {s_tensor.sum()}")

    tracks = []
    node_idx = 0
    for track_idx in range(s_tensor.size(0)):
        print(f"Processing track idx: {track_idx}")
        track_structure = []
        for node_idx, idx in enumerate(range(s_tensor.size(-1))):
            print(f"Processing phrase {idx} in track {track_idx}")

            print(f"node_idx: {node_idx}")

            if (
                s_tensor[track_idx, :, :, idx].sum() == 0
            ):  # If there's no activation in Mel/Non-Mel
                # Skip this note
                print("No activation, skipping phrase")
                continue

            # print(f"Melody activation tensor: {s_tensor[track_idx, 0, :, idx]}")
            # print(f"Melody activation tensor: {s_tensor[track_idx, 0, :, idx].nonzero()[0]}")
            mel_act = s_tensor[track_idx, 0, :, idx].nonzero()[0]
            # print(f"Melody activation index: {mel_act}")

            phrase_ids = c_logits[node_idx, 0, : constants.N_ID_TOKENS]
            phrase_lens = c_logits[node_idx, 0, constants.N_ID_TOKENS :]
            # print(f"Phrase id: {phrase_ids.shape}, phrase len: {phrase_lens.shape}")
            phrase_ids, phrase_lens = (
                torch.argmax(phrase_ids),
                torch.argmax(phrase_lens),
            )
            print(f"Phrase id: {phrase_ids}, phrase len: {phrase_lens}")

            #             if (phrase_ids == IdToken.EOS.value or
            #                 phrase_ids == IdToken.PAD.value or
            #                 phrase_lens == LengthToken.EOS.value or
            #                 phrase_lens == LengthToken.PAD.value):
            #                 # The phrase contains no additional notes, go to next chord

            #                 print("No additional phrases, skipping track")

            #                 break

            if phrase_ids == IdToken.SOS.value or phrase_ids == IdToken.SOS.value:
                # Skip this note
                print("SOS, skipping phrase")
                continue

            # Remapping duration values from [0, 95] to [1, 96]
            # dur = dur + 1
            # Do not sustain notes beyond sequence limit
            # dur = min(dur.item(), mtp.size(1) - t)

            track_structure.append(
                {
                    "phrase_id": phrase_ids,
                    "length": phrase_lens,
                    "is_melody": bool(mel_act),
                }
            )

        print(track_structure)
        tracks.append(track_structure)
    return tracks, s_tensor


# def save(mtp, dir, s_tensor=None, n_loops=1, audio=True,
#          looped_only=False, plot_proll=False, plot_struct=False):

#     n_bars = mtp.size(1)
#     resolution = mtp.size(3) // 4
#     # Clear matplotlib cache (this solves formatting problems with first plot)
#     plt.clf()

#     # Iterate over batches
#     for i in range(mtp.size(0)):

#         # Create the directory if it does not exist
#         save_dir = os.path.join(dir, str(i))
#         os.makedirs(save_dir, exist_ok=True)

#         if not looped_only:
#             # Generate MIDI song from multitrack pianoroll and save
#             muspy_song = muspy_from_mtp(mtp[i])
#             print("Saving MIDI sequence {} in {}...".format(str(i + 1),
#                                                             save_dir))
#             save_midi(muspy_song, save_dir, name='generated')
#             if audio:
#                 print("Saving audio sequence {} in {}...".format(str(i + 1),
#                                                                  save_dir))
#                 save_audio(muspy_song, save_dir, name='generated')

#         if plot_proll:
#             plot_pianoroll(muspy_song, save_dir)

#         if plot_struct:
#             print(s_tensor[i].shape)
#             plot_structure(s_tensor[i].cpu(), save_dir)

#         if n_loops > 1:
#             # Copy the generated sequence n_loops times and save the looped
#             # MIDI and audio files
#             print("Saving MIDI sequence "
#                   "{} looped {} times in {}...".format(str(i + 1), n_loops,
#                                                        save_dir))
#             extended = loop_muspy_music(muspy_song, n_loops,
#                                          n_bars, resolution)
#             save_midi(extended, save_dir, name='extended')
#             if audio:
#                 print("Saving audio sequence "
#                       "{} looped {} times in {}...".format(str(i + 1), n_loops,
#                                                            save_dir))
#                 save_audio(extended, save_dir, name='extended')

#         print()


def generate_z(bs, d_model, device):
    shape = (bs, d_model)
    z_norm = torch.normal(
        torch.zeros(shape, device=device), torch.ones(shape, device=device)
    )
    return z_norm


def load_model(model_dir, device):
    checkpoint = torch.load(os.path.join(model_dir, "checkpoint"), map_location="cpu")
    configuration = torch.load(
        os.path.join(model_dir, "configuration"), map_location="cpu"
    )

    state_dict = checkpoint["model_state_dict"]

    model = VAE(**configuration["model"], device=device).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, configuration


def main():
    parser = argparse.ArgumentParser(
        description="Generates MIDI music with a trained model."
    )
    parser.add_argument("model_dir", type=str, help="Directory of the model.")
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the generated MIDI files."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of sequences to be generated. Default is 5.",
    )
    parser.add_argument(
        "--s_file",
        type=str,
        help="Path to the JSON file containing the binary structure tensor.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Flag to enable GPU usage.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default="0",
        help="Index of the GPU to be used. Default is 0.",
    )
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    # audio = not args.no_audio

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    if args.use_gpu:
        torch.cuda.set_device(args.gpu_id)

    print_divider()
    print("Loading the model on {} device...".format(device))

    model, configuration = load_model(args.model_dir, device)

    print(f"Model config: {configuration['model']}")

    d_model = configuration["model"]["d"]
    n_bars = configuration["model"]["n_bars"]
    n_tracks = constants.N_TYPES
    n_timesteps = 4 * configuration["model"]["resolution"]
    output_dir = args.output_dir

    s, s_tensor = None, None

    #     if args.s_file is not None:

    #         print("Loading the structure tensor "
    #               "from {}...".format(args.model_dir))

    #         # Load structure tensor from file
    #         with open(args.s_file, 'r') as f:
    #             s_tensor = json.load(f)

    #         s_tensor = torch.tensor(s_tensor, dtype=bool)

    #         # Check structure dimensions
    #         dims = list(s_tensor.size())
    #         expected = [n_bars, n_tracks, n_timesteps]
    #         if dims != expected:
    #             if (len(dims) != len(expected) or dims[1:] != expected[1:]
    #                     or dims[0] > n_bars):
    #                 raise ValueError(f"Loaded structure tensor dimensions {dims} "
    #                                  f"do not match expected dimensions {expected}")
    #             elif dims[0] > n_bars:
    #                 raise ValueError(f"First structure tensor dimension {dims[0]} "
    #                                  f"is higher than {n_bars}")
    #             else:
    #                 # Repeat partial structure tensor
    #                 r = math.ceil(n_bars / dims[0])
    #                 s_tensor = s_tensor.repeat(r, 1, 1)
    #                 s_tensor = s_tensor[:n_bars, ...]

    #         # Avoid empty bars by creating a fake activation for each empty
    #         # (n_tracks x n_timesteps) bar matrix in position [0, 0]
    #         empty_mask = ~s_tensor.any(dim=-1).any(dim=-1)
    #         if empty_mask.any():
    #             print("The provided structure tensor contains empty bars. Fake "
    #                   "track activations will be created to avoid processing "
    #                   "empty bars.")
    #         idxs = torch.nonzero(empty_mask, as_tuple=True)
    #         s_tensor[idxs + (0, 0)] = True

    #         # Repeat structure along new batch dimension
    #         s_tensor = s_tensor.unsqueeze(0).repeat(args.n, 1, 1, 1)

    #         s = model.decoder._structure_from_binary(s_tensor)

    print()
    print("Generating z...")
    z = generate_z(args.n, d_model, device)

    print("Generating structures with the model...")
    s_t = time.time()
    structure = generate_structure(model, z, s, s_tensor)
    print("Inference time: {:.3f} s".format(time.time() - s_t))

    # print()
    # print("Saving structure files in {}...\n".format(output_dir))
    # save(structure, output_dir)
    # print("Finished saving structure files.")
    print_divider()


if __name__ == "__main__":
    main()
