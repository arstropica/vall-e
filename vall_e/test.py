import logging
from collections import defaultdict

from functools import cache, partial
from pathlib import Path
import shutil
import uuid

import argparse

import torch
import torchaudio

from einops import rearrange, repeat
from .data import _load_quants

from .config import cfg
from .emb import qnt
from .emb import g2p
from .utils import setup_logging, to_device, trainer, gather_attribute
from .vall_e import get_model

from .vall_e.nar import example_usage
import soundfile

_logger = logging.getLogger(__name__)
_temp_path: Path
_file_path : Path
_text_path: Path
_device = "cuda"

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--suffix", type=str, default=".normalized.txt")
    args = parser.parse_args()
    file_path = args.file
    if not file_path.exists():
        raise RuntimeError(f"Failed to find {args.file.absolute()} file.")
    return args

def setup():
    global _temp_path, _file_path, _text_path
    args = _args()
    file_path = args.file.absolute()
    text_path = file_path.with_name(file_path.stem.split(".")[0] + args.suffix)
    cwd = Path(file_path).parent
    _temp_path = Path(str(cwd) + "/" + str(uuid.uuid1()) + "/")

    if text_path.exists():
        _temp_path.mkdir(parents=True, exist_ok=True)
        _file_path = Path(shutil.copy(file_path, _temp_path))
        _text_path = Path(shutil.copy(text_path, _temp_path))
    else:
        cleanup()
        raise RuntimeError(f"Failed to find {text_path.absolute()} file.")

def cleanup():
    if _temp_path and _temp_path.exists():
        try:
            shutil.rmtree(_temp_path)
        except OSError as e:
            print("Error: %s : %s" % (_temp_path, e.strerror))

def quantize():
    file_path = _file_path
    qnt_path = qnt._replace_file_extension(file_path, ".qnt.pt")
    if qnt_path.exists():
        return qnt_path
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] == 2:
        wav = wav[:1]
    encoded = qnt.encode(wav, sr)
    torch.save(encoded.cpu(), qnt_path)
    return qnt_path

def get_text():

    def _get_phone_path():
        return _file_path.with_name(_file_path.stem.split(".")[0] + ".phn.txt")

    def _get_phones():
        phone_path = _get_phone_path()
        #if phone_path.exists():
        #    return phone_path
        
        graphs = g2p._get_graphs(_text_path)
        phones = g2p.encode(graphs)
        with open(phone_path, "w") as f:
            f.write(" ".join(phones))

        return ["<s>"] + phones + ["</s>"]
        
    def _phones():
        path = _get_phone_path()
        return sorted(set().union(*[_get_phones() for path in [path]]))

    def _get_phone_symmap(*args):
        # Note that we use phone symmap starting from 1 so that we can safely pad 0.
        return {s: i for i, s in enumerate(_phones(), 1)}

    def _data():
        phone_symmap = _get_phone_symmap()
        text = torch.tensor([*map(phone_symmap.get, _get_phones())], device=_device)
        return text
    
    return _data()

def get_prompts():
    path = quantize()
    prom_list = [_load_quants(path)]
    return torch.cat(prom_list).to(device=_device)

def load_engines():
    model = get_model(cfg.model)

    engines = dict(
        model=trainer.Engine(
            model=model,
            config=cfg.ds_cfg,
        ),
    )

    return trainer.load_engines(engines, cfg)

def main():
    setup_logging(cfg.log_dir)

    @torch.inference_mode()
    def eval():
        args = _args()
        out_dir = args.file.parent
        filename = args.file.stem
        out_file = Path(str(out_dir.absolute()) + "/" + str(filename) + ".nar.wav").with_suffix(".wav")

        setup()

        text_elem = get_text()
        prom_elem = get_prompts()
        resps = torch.load(quantize())[0].to(_device)
        text_list = [text_elem]
        proms_list = [prom_elem]
        resp_list = [resps[0].to(_device)]

        resps_list = [resps.t().to(_device)]

        engines = load_engines()
        engines.eval()
        model = engines["model"]

        if cfg.model.startswith("ar"):
            out = model(
                text_list=text_list,
                proms_list=proms_list,
                max_steps=cfg.max_val_ar_steps,
            )
            out = [r.unsqueeze(-1) for r in out]
        elif cfg.model.startswith("nar"):
            out = model(
                text_list=text_list,
                proms_list=proms_list,
                resp_list=resp_list,
            )
        else:
            raise NotImplementedError(cfg.model)
        
        qnt.decode_to_file(out[0], out_file)
        qnt.unload_model()
        cleanup()
    
    eval()

def example():
    example_usage()

if __name__ == "__main__":
    main()
    #example()
