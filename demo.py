# -*- coding: utf-8 -*-
# @Time    : 5/8/23
# @Author  : Yaojie Shen
# @Project : Affective-Computing-Demo
# @File    : demo.py


import argparse
import os

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

import librosa
import time

from mme2e.models.e2e import MME2E
from mme2e.datasets import getEmotionDict

EMOTION_DICT = {
    'ang': "Angry", 'exc': "Excited", 'fru': "Frustrated", 'hap': "Happy", 'neu': "Neutral", 'sad': "Sad"
}


class StreamingDemoRunner(object):
    def __init__(self, model):
        self.model = model

        self.image_queue = []
        self.audio_queue = None
        self.sr = None

        self.idx2label = {v: k for k, v in getEmotionDict().items()}
        self.label_annotations = [self.idx2label[i] for i in range(len(self.idx2label))]

    def cutSpecToPieces(self, spec, stride=32):
        # Split the audio waveform by second
        total = -(-spec.size(-1) // stride)
        specs = []
        for i in range(total):
            specs.append(spec[:, :, :, i * stride:(i + 1) * stride])

        # Pad the last piece
        lastPieceLength = specs[-1].size(-1)
        if lastPieceLength < stride:
            padRight = stride - lastPieceLength
            specs[-1] = F.pad(specs[-1], (0, padRight))
        return specs

    @staticmethod
    def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or "Spectrogram (db)")
        axs.set_ylabel(ylabel)
        axs.set_xlabel("frame")
        im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
        fig.colorbar(im, ax=axs)
        # plt.show(block=False)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    @torch.no_grad()
    def run(self, image, audio):
        if image is not None:
            self.image_queue.append(image)
            self.image_queue = self.image_queue[-4:]
        if audio is not None:
            sr, waveform = audio
            self.sr = sr
            self.audio_queue = np.concatenate([self.audio_queue, waveform], axis=0) \
                if self.audio_queue is not None else waveform
            self.audio_queue = self.audio_queue[-sr * 10:, ...]

        if len(self.image_queue) == 4 and (audio is not None or 'a' not in self.model.mod):

            if 'v' in self.model.mod:
                video_clip = torch.from_numpy(np.stack(self.image_queue)).to(torch.float32)  # sum(N) H W C
            else:
                video_clip = None

            if 'a' in self.model.mod:
                sr, waveform = self.sr, self.audio_queue
                print(f"Audio length: {len(waveform) / sr}")
                print(f"Sample rate: {sr}")
                if len(waveform.shape) == 2:
                    waveform = waveform[:, 0]
                waveform = torch.from_numpy(np.expand_dims(waveform, axis=0)).to(torch.float32)

                specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform).unsqueeze(0)
                specgrams = self.cutSpecToPieces(specgram)
                spec_lens = [len(specgrams)]
                specgrams = torch.cat(specgrams, dim=0)
            else:
                specgrams = None
                spec_lens = None

            pred = self.model(imgs=video_clip, imgs_lens=[4], specs=specgrams, spec_lens=spec_lens, text=None)
            pred = F.softmax(pred, dim=-1)[0].detach().cpu().numpy().tolist()
            return {EMOTION_DICT[self.label_annotations[i]]: v for i, v in enumerate(pred)}, \
                self.plot_spectrogram(specgram[0, 0, ...]) if 'a' in self.model.mod else None
        else:
            print(f"Data not enough, skip.")
            return {EMOTION_DICT[k]: .0 for k in self.label_annotations}, None


def find_checkpoint(model_path, model_name, modalities):
    for model in os.listdir(model_path):
        if len(model.split("_")) < 2 or not model.endswith(".pt"):
            continue
        name, mod = model.split("_")[:2]
        if name == model_name and set(list(modalities)) == set(list(mod)):
            return os.path.join(model_path, model)
    raise FileNotFoundError(f"Cannot found model in {model_path}, require {model_name}_{modalities}_xxx.pt")


def main(args):
    device = torch.device(args.device)

    if args.model == "mme2e":
        model_args = vars(args)
        model = MME2E(args=model_args, device=device)

        state_dict = torch.load(find_checkpoint(args.model_zoo, args.model, args.modalities), map_location="cpu")
        model.load_state_dict(state_dict)

        model = model.to(device=device)
    else:
        raise ValueError(f"Model type '{args.model}' is not supported.")

    runner = StreamingDemoRunner(model)

    demo = gr.Interface(
        fn=runner.run,
        inputs=[
            gr.Image(source="webcam", streaming=True, shape=(360, 360)),
            gr.Audio(source="microphone", streaming=True, )
        ],
        outputs=[
            gr.outputs.Label(type="confidences", label="Emotion"),
            gr.outputs.Image("numpy", label="MelSpectrogram")
        ],
        live=True,
        allow_flagging="never"
    )
    demo.launch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run demo")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--model_zoo", help="Model directory", type=str, required=True)

    # model
    parser.add_argument("--model", choices=["mme2e"], default="mme2e")

    parser.add_argument('--text-model-size', help='Size of the pre-trained text model', type=str, required=False,
                        default='base')

    parser.add_argument('--feature-dim', help='Dimension of features outputed by each modality model', type=int,
                        required=False, default=256)
    parser.add_argument('--trans-dim', help='Dimension of the transformer after CNN', type=int, required=False,
                        default=512)
    parser.add_argument('--trans-nlayers', help='Number of layers of the transformer after CNN', type=int,
                        required=False, default=2)
    parser.add_argument('--trans-nheads', help='Number of heads of the transformer after CNN', type=int, required=False,
                        default=8)

    parser.add_argument('--num-emotions', help='Number of emotions in data', type=int, required=False, default=4)
    parser.add_argument('-mod', '--modalities', help='what modalities to use', type=str, required=False, default='tav')

    main(parser.parse_args())
