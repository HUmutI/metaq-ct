"""Build the official fVLM model and load the official CT-RATE checkpoint.

Copied/adapted from fvlm/lavis/models/blip_models/blip_pretrain.py::from_config.
CHANGES vs the repo (documented, no architecture change):
  1. The repo's from_config() torch.load()s an ImageNet MAE ViT checkpoint from a
     hardcoded path that does not exist here ('/storage/guoruizhe/.../mae_pretrain_vit_base.pth').
     That load is only an INITIALIZATION; every visual_encoder parameter is
     subsequently overwritten by the released CT-RATE model.pth (verified: all 149
     visual_encoder tensors are present in the checkpoint and match shapes).
     We therefore skip it and load model.pth instead.
  2. cfg is a plain dict instead of a lavis omegaconf Config (omegaconf not installed).
Architecture (ViT dims, patch size, organs, projections, attention) is untouched.
"""
import os, sys, torch

FVLM_REPO = '/home/ch278233/BENCHMARK/fvlm'
CKPT = '/temp_work/ch278233/BENCHMARK_WEIGHTS/fvlm/checkpoints/model.pth'
RUNDIR = '/temp_work/ch278233/PEDS_BENCH/fvlm/run'   # must contain BiomedVLP-CXR-BERT-specialized/

class _Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)

def build_model(device='cuda', verbose=True):
    os.chdir(RUNDIR)                     # repo resolves BiomedVLP-CXR-BERT-specialized relatively
    if FVLM_REPO not in sys.path:
        sys.path.insert(0, FVLM_REPO)

    import lavis.models.med as med
    # med_config_path is resolved via get_abs_path() then .replace('lavis','lavis/..');
    # point it at our local copy (path contains no substring 'lavis').
    med.get_abs_path = lambda rel: os.path.join(RUNDIR, 'BiomedVLP-CXR-BERT-specialized/config.json')

    from lavis.models.blip_models.vit import ViT
    from lavis.models.med import XBertEncoder
    from lavis.models.blip_models.blip_pretrain import BlipPretrain

    cfg = _Cfg(med_config_path='BiomedVLP-CXR-BERT-specialized/config.json',
               alpha=0.5, max_txt_len=384)

    image_encoder = ViT(in_channels=1, img_size=(112, 256, 352), patch_size=(16, 16, 32),
                        num_classes=0, dropout_rate=0.1, qkv_bias=True)
    text_encoder = XBertEncoder.from_config(cfg, from_pretrained=True)

    model = BlipPretrain(image_encoder=image_encoder, text_encoder=text_encoder,
                         text_decoder=None, embed_dim=256, alpha=0.5,
                         tie_enc_dec_weights=False, max_txt_len=384)

    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    msd = model.state_dict()
    matched = [k for k in sd if k in msd and msd[k].shape == sd[k].shape]
    shape_mismatch = [(k, tuple(sd[k].shape), tuple(msd[k].shape)) for k in sd
                      if k in msd and msd[k].shape != sd[k].shape]
    unexpected = [k for k in sd if k not in msd]
    missing = [k for k in msd if k not in sd]
    info = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f'[ckpt] tensors in checkpoint : {len(sd)}')
        print(f'[ckpt] tensors in model      : {len(msd)}')
        print(f'[ckpt] MATCHED & LOADED      : {len(matched)}')
        print(f'[ckpt] shape mismatches      : {len(shape_mismatch)} {shape_mismatch[:5]}')
        print(f'[ckpt] unexpected (ckpt only): {len(unexpected)} {unexpected[:10]}')
        print(f'[ckpt] missing  (model only) : {len(missing)} {missing[:10]}')
        vis = [k for k in matched if k.startswith('visual_encoder')]
        txt = [k for k in matched if k.startswith('text_encoder')]
        print(f'[ckpt]   of which visual_encoder={len(vis)} text_encoder={len(txt)} '
              f'heads={len(matched)-len(vis)-len(txt)}')
        print(f'[ckpt] temp={model.temp.item():.6f}  organs={model.organs}')
    model.eval().to(device)
    return model, dict(n_ckpt=len(sd), n_model=len(msd), matched=len(matched),
                       unexpected=unexpected, missing=missing,
                       shape_mismatch=shape_mismatch)

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m, info = build_model(device=dev)
    print('built ok on', dev)
