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
import matplotlib as mpl
import matplotlib.style as mplstyle

import librosa
import time

from mme2e.models.e2e import MME2E
from mme2e.datasets import getEmotionDict

import logging
import warnings

warnings.filterwarnings("ignore")

EMOTION_DICT = {
    'ang': "Angry", 'exc': "Excited", 'fru': "Frustrated", 'hap': "Happy", 'neu': "Neutral", 'sad': "Sad"
}

logger = logging.getLogger("web_demo")


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
    def plot_spectrogram(specgram):
        mpl.rcParams['path.simplify'] = True
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mplstyle.use('fast')

        fig = plt.figure(dpi=specgram.shape[0])
        fig.set_size_inches(4, 1, forward=False)
        axs = plt.Axes(fig, [0., 0., 1., 1.])
        axs.set_axis_off()
        fig.add_axes(axs)
        axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.clf()
        return data

    @staticmethod
    def plot_wave(sr, waveform):
        mpl.rcParams['path.simplify'] = True
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mplstyle.use('fast')

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sr
        fig = plt.figure(dpi=150)
        fig.set_size_inches(4, 1, forward=False)
        axs = plt.Axes(fig, [0., 0., 1., 1.])
        axs.set_axis_off()
        fig.add_axes(axs)
        axs.plot(time_axis, waveform[0], linewidth=0.5)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.clf()
        return data

    @torch.no_grad()
    def run(self, image, audio):
        if image is not None:
            self.image_queue.append(image)
            self.image_queue = self.image_queue[-4:]
        if audio is not None:
            sr, waveform = audio
            self.sr = sr
            try:
                self.audio_queue = np.concatenate([self.audio_queue, waveform], axis=0)
            except ValueError:
                self.audio_queue = waveform  # init
            self.audio_queue = self.audio_queue[-sr * 10:, ...]

        if len(self.image_queue) == 4 and (audio is not None or 'a' not in self.model.mod):

            if 'v' in self.model.mod:
                video_clip = np.stack(self.image_queue).astype(float)  # sum(N) H W C
            else:
                video_clip = None

            if 'a' in self.model.mod:
                sr, waveform = self.sr, self.audio_queue
                logger.debug(f"Audio length: {len(waveform) / sr}")
                logger.debug(f"Sample rate: {sr}")
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
                self.plot_spectrogram(specgram[0, 0, ...]) if 'a' in self.model.mod else None, \
                self.plot_wave(sr, waveform) if 'a' in self.model.mod else None
        else:
            logger.warning(f"Data not enough, skip.")
            return {EMOTION_DICT[k]: .0 for k in self.label_annotations}, None, None


def find_checkpoint(model_path, model_name, modalities):
    model_list = []
    for model in os.listdir(model_path):
        try:
            if len(model.split("_")) < 2 or not model.endswith(".pt"):
                continue
            name, mod, _, acc = model.split("_")[:4]
            model_list.append((model, name, mod, float(acc)))
        except Exception as e:
            logger.error(f"Cannot parse model file '{model}'")
            raise e
    model_list.sort(key=lambda x: x[-1], reverse=True)
    for model, name, mod, acc in model_list:
        if name == model_name and set(list(modalities)) == set(list(mod)):
            model_path = os.path.join(model_path, model)
            logger.info(f"Find trained model '{model_path}'.")
            return model_path
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
            gr.Image(source="webcam", streaming=True, shape=(224, 224)),
            gr.Audio(source="microphone", streaming=True)
        ],
        outputs=[
            gr.outputs.Label(type="confidences", label="Emotion"),
            gr.outputs.Image("numpy", label="MelSpectrogram"),
            gr.outputs.Image("numpy", label="Waveform")
        ],
        live=True,
        allow_flagging="never",
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

    # setup logger
    # logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter(
        f"[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.DEBUG)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)
    logger.propagate = False

    main(parser.parse_args())
