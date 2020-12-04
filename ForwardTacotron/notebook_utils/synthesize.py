import torch

from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from models.tacotron import Tacotron
from utils.text.symbols import phonemes
from utils.text import text_to_sequence, clean_text
from utils.dsp import reconstruct_waveform
from utils import hparams as hp


def init_hparams(hp_file):
    hp.configure(hp_file)


def get_forward_model(model_path, device=torch.device('cuda')):
    #device = torch.device('cuda')
    model = ForwardTacotron(embed_dims=hp.forward_embed_dims,
                            num_chars=len(phonemes),
                            durpred_rnn_dims=hp.forward_durpred_rnn_dims,
                            durpred_conv_dims=hp.forward_durpred_conv_dims,
                            durpred_dropout=hp.forward_durpred_dropout,
                            rnn_dim=hp.forward_rnn_dims,
                            postnet_k=hp.forward_postnet_K,
                            postnet_dims=hp.forward_postnet_dims,
                            prenet_k=hp.forward_prenet_K,
                            prenet_dims=hp.forward_prenet_dims,
                            highways=hp.forward_num_highways,
                            dropout=hp.forward_dropout,
                            n_mels=hp.num_mels).to(device)
    model.load(model_path)
    return model


def get_wavernn_model(model_path):
    device = torch.device('cuda')
    print()
    model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode=hp.voc_mode).to(device)

    model.load(model_path)
    return model


def get_melgan_model():
    vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
    vocoder.cuda().eval()
    return vocoder

def get_tacotron_model(model_path):
    device = torch.device('cuda')
    model = Tacotron(embed_dims=hp.tts_embed_dims,
                    num_chars=len(phonemes),
                    encoder_dims=hp.tts_encoder_dims,
                    decoder_dims=hp.tts_decoder_dims,
                    n_mels=hp.num_mels,
                    fft_bins=hp.num_mels,
                    postnet_dims=hp.tts_postnet_dims,
                    encoder_K=hp.tts_encoder_K,
                    lstm_dims=hp.tts_lstm_dims,
                    postnet_K=hp.tts_postnet_K,
                    num_highways=hp.tts_num_highways,
                    dropout=hp.tts_dropout,
                    stop_threshold=hp.tts_stop_threshold).to(device)
    model.load(model_path)
    return model


def synthesize(input_text, tts_model, voc_model, alpha=1.0, device=torch.device('cuda')):
    text = clean_text(input_text.strip())
    x = text_to_sequence(text)
    _, m, _ = tts_model.generate(x, alpha=alpha)
    if voc_model == 'griffinlim':
        wav = reconstruct_waveform(m, n_iter=32)
    elif isinstance(voc_model, WaveRNN):
        m = torch.tensor(m).unsqueeze(0)
        wav = voc_model.generate(m, '/tmp/sample.wav', True, hp.voc_target, hp.voc_overlap, hp.mu_law)
    else:
        m = torch.tensor(m).unsqueeze(0).to(device)
        with torch.no_grad():
            wav = voc_model.inference(m).cpu().numpy()
    return wav

