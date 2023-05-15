# -*- coding: utf-8 -*-
# @Time    : 5/14/23
# @Author  : Yaojie Shen
# @Project : Affective-Computing-Demo
# @File    : build_video.py

import glob
import argparse
import os

import moviepy.editor


def image2video(image_list, audio_file=None, output_file="output.mp4"):
    if audio_file is not None:
        audio = moviepy.editor.CompositeAudioClip([moviepy.editor.AudioFileClip(audio_file)])
        fps = int(len(image_list) / audio.duration)
    else:
        audio = None
        fps = 30

    video = moviepy.editor.ImageSequenceClip(image_list, fps=fps, )

    video = video.set_audio(audio)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    video.write_videofile(output_file, codec='libx264',
                          audio_codec='aac',
                          temp_audiofile='temp-audio.m4a',
                          remove_temp=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image + Audio -> Video")
    parser.add_argument("image_file_pattern", type=str)
    parser.add_argument("--audio_file", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default="output.mp4")

    args = parser.parse_args()
    file_list = glob.glob(args.image_file_pattern)
    file_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    print("Images:" + "".join(["\t" + str(img) + "\n" for img in file_list]) + f"Audio: {args.audio_file}")

    image2video(file_list, args.audio_file, args.output)
