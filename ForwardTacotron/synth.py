from notebook_utils.synthesize import (
    get_forward_model, get_melgan_model,
    get_wavernn_model, get_tacotron_model,
    synthesize, init_hparams)
import torch
from model.generator import Generator
from utils import hparams as hp
import librosa
import argparse
import numpy as np
import time

init_hparams('hparams.py')

parser = argparse.ArgumentParser(description='Start TTS and Vocoder')
parser.add_argument('--path_taco', default='checkpoints/ljspeech_tts.forward/latest_weights.pyt')
parser.add_argument('--path_vocoder', default='None')
parser.add_argument('--device', default='cuda')
parser.add_argument('--sentense', default='ŦīN šŦRīY SūRԞā Kē ǤēRē Mē JāŋC KRāNē Kē LīE KŦāR Mē ĶDā HōNā PDā')
parser.add_argument('--file_name', default='test')
parser.add_argument('--alpha', default=1)
args = parser.parse_args()

print('get:\n--path: ' + args.path_taco +
      '\n--path_vocoder' + args.path_vocoder +
      '\n--sentense: ' + args.sentense +
      '\n--file_name: ' + args.file_name +
      '\n--alpha: ' + str(args.alpha) +
      '\n--device:' + args.device)
if args.device == 'cuda':
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
tts_model = get_forward_model(args.path_taco, device=device)
print('tts loaded')
if args.path_vocoder == 'None':
    voc_model = get_melgan_model().to(device)
else:
    checkpoint = torch.load(args.path_vocoder)
    #hp = load_hparam_str(checkpoint['hp_str'])
    voc_model = Generator(80).to(device)
    voc_model.load_state_dict(checkpoint['model_g'])
    voc_model.eval(inference=False)
print('MelGan loaded')
now = time.time()
wav = synthesize(args.sentense, tts_model, voc_model, alpha=float(args.alpha), device=device)
synt = time.time()
wav = wav.astype(np.float32, order='C') / 32768.0
convert = time.time()
librosa.output.write_wav(args.file_name + '.wav', wav, 22050)
write = time.time()
print('\nsynthesize: ', synt - now)
print('convert: ', convert - synt)
print('write: ', write - synt)
print('total: ', write - now)
print(args.file_name + '.wav is writen')

