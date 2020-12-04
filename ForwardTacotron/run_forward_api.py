
#from wavenet_vocoder.synthesize import wavenet_synthesize
import os
from flask import Flask
from flask import send_file
from flask import request

import tensorflow as tf
import librosa
import argparse
import yaml
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor

app = Flask(__name__)

@app.route('/tts/')
def start_tts():
    id = request.args.get("ID","")
    text = request.args.get("text","")
    alpha = request.args.get("alpha", 1)
    #print("id = {} | text = {} | alpha = {} | time = {}".format(id, text, alpha))
    #sentense = text.replace("%", " ").replace(", TORCHA, ", " ")
    try:
        now = time.time()
        #wav = synthesize(sentense, taco_model, voc_model, alpha=float(alpha), device=device)
        mels, audios = do_synthesis(args.sentense, fastspeech2, mb_melgan, "FASTSPEECH2", "MB-MELGAN")
        #wav = wav.astype(np.float32, order='C') / 32768.0
        wav = audios.astype(np.float32, order='C') #/ 32768.0
        sf.write('audio/' + id + '.wav', wav, 22050)
        print("id = {} | text = {} | alpha = {} | time = {}".format(id, text, alpha, time.time()-now))
        return send_file('audio/{}.wav'.format(id), attachment_filename='{}.wav'.format(id))
    except Exception as e:
        return str(e)

def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
    input_ids = processor.text_to_sequence(input_text)

    # text2mel part
    if text2mel_name == "TACOTRON":
        _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )
    elif text2mel_name == "FASTSPEECH":
        mel_before, mel_outputs, duration_outputs = text2mel_model.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
    elif text2mel_name == "FASTSPEECH2":
        mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
    else:
        raise ValueError("Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name")

    # vocoder part
    if vocoder_name == "MELGAN" or vocoder_name == "MELGAN-STFT":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    elif vocoder_name == "MB-MELGAN":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    else:
        raise ValueError("Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name")

    if text2mel_name == "TACOTRON":
        return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
    else:
        return mel_outputs.numpy(), audio.numpy()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start TTS and Vocoder')
    parser.add_argument('--path_fs', default="examples/fastspeech2_libritts/outdir_libri/checkpoints/model-855000.h5")
    parser.add_argument('--path_mb', default="checks/mb_melgan_or/mb.melgan-940k.h5")

    args = parser.parse_args()

    fastspeech2_config = AutoConfig.from_pretrained('examples/fastspeech2/conf/fastspeech2.v1.yaml')
    fastspeech2 = TFAutoModel.from_pretrained(
        config=fastspeech2_config,
        pretrained_path= args.path_fs, #"examples/fastspeech2_libritts/outdir_libri/checkpoints/model-855000.h5",
        #training=False,
        name="fastspeech2"
    )

    mb_melgan_config = AutoConfig.from_pretrained('examples/multiband_melgan/conf/multiband_melgan.v1.yaml')
    mb_melgan = TFAutoModel.from_pretrained(
        config=mb_melgan_config,
        pretrained_path= args.path_mb, #"checks/mb_melgan_or/mb.melgan-940k.h5",
        name="mb_melgan"
    )

    processor = AutoProcessor.from_pretrained(pretrained_path="dump_ljspeech/ljspeech_mapper.json")

    app.run(host = '0.0.0.0',port=5454)
