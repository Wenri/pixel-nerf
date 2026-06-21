# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

pixelNeRF (CVPR 2021): a NeRF conditioned on pixel-aligned image features, for novel view synthesis from one or few input views. Research codebase, no test suite or linter config. PyTorch 1.6.0 / CUDA 10.2 era.

## Environment

```sh
conda env create -f environment.yml && conda activate pixelnerf   # or: pip install -r requirements.txt
```

## Running things

**Always run scripts from the project root.** Config paths (`conf/exp/...`), `expconf.conf` discovery, and HOCON `include required("../default_mv.conf")` are all resolved relative to the repo root / conf-file location.

- **Train:** `python train/train.py -n <exp> -c conf/exp/<exp>.conf -D <data_root> -V <nviews> --gpu_id=<id> --resume`
- **Full eval:** `python eval/eval.py -D <data_root> -n <exp> -L viewlist/src_*.txt -O eval_out/<name>` (resume-capable — rerun to continue; parallelized over GPUs)
- **Final metrics (PSNR/SSIM/LPIPS):** `python eval/calc_metrics.py ...` — run after `eval.py`; reads rendered images from the `-O` dir. Has its own standalone arg parser (does not use `util/args.py`).
- **Approximate eval (fast):** `python eval/eval_approx.py -D <data> -n <exp>`
- **Video / demo:** `python eval/gen_video.py ...` → `visuals/<exp>/...mp4`; `eval/eval_real.py` for real car images (needs detectron2 PointRend + `scripts/preproc.py` first)
- **TensorBoard:** `tensorboard --logdir logs/<expname>`; visualizations also dumped to `visuals/<exp>/<epoch>_<batch>_vis.png` (rows: coarse/fine; cols: input views, depth, output, alpha)

### Shared CLI flags (`src/util/args.py`)
All training/eval scripts share one parser. Key flags: `-n` expname, `-c` config, `-D` datadir, `-F` dataset format, `--split`, `-S` scene/object id, `-P '<v1 v2>'` fixed source views (or `-L` viewlist file), `--gpu_id='0 1 3'` (space-delimited), `-R` ray batch size (lower if OOM: default 50000 eval / 128 train).

**Expname auto-inference:** `expconf.conf` maps each `-n <expname>` to a default `-c` config and `-D` datadir. For the provided experiments (`sn64`, `sn64_unseen`, `srn_chair`, `srn_car`, `dtu`, `multi_obj`) you can omit `-c`. For new expnames, either pass `-c` explicitly or add a row to `expconf.conf`.

## Architecture

### Import layout (important)
Entry-point scripts in `train/` and `eval/` do `sys.path.insert(0, .../src)`, so code imports the `src/` packages by bare name: `import util`, `from model import ...`, `from render import ...`, `from data import ...`. There is no `src.` prefix. Inside `src/`, modules likewise import each other by bare name (e.g. `import util` from within `model/`).

### Inference pipeline
1. `data/` → `get_split_dataset(format, datadir)` returns dataset objects. `z_near`/`z_far`/`lindisp` are **dataset attributes**, read off the dataset and passed into the renderer/ray generation — not hardcoded.
2. `model/PixelNeRFNet` (`src/model/models.py`) — the core. Two phases:
   - `.encode(images, poses, focal, c)`: runs the CNN encoder, caches the feature volume, and stores **world→camera** poses, focal (note `focal[...,1] *= -1`), and principal point. Source views are the `NS` dimension.
   - `.forward(xyz)`: transforms query points into each source view's camera frame, projects to pixel coords (`uv = -xyz[:,:2]/xyz[:,2:]`, OpenGL-style −z camera), samples pixel-aligned features via `encoder.index(uv)` (bilinear `grid_sample`), concatenates with positionally-encoded xyz + view dirs, and runs the MLP. Returns `(SB, B, 4)` = rgb + sigma.
3. `model/encoder.py` — `SpatialEncoder` (pixel-aligned, default `resnet34` truncated to `num_layers`, multi-scale features upsampled and concatenated) is the main one. `ImageEncoder` is an optional global feature vector.
4. `model/resnetfc.py::ResnetFC` — the NeRF MLP. **Multi-view aggregation happens inside this MLP**, not before it: features for the `NS` source views flow through independently until `combine_layer`, where `util.combine_interleaved` reduces over the source-view dimension (`combine_type` = average | max). The latent is injected per-block via `lin_z`.
5. `render/nerf.py::NeRFRenderer` — hierarchical coarse/fine sampling + alpha compositing; consumes rays `[origin(3), dir(3), near(1), far(1)]`. `bind_parallel(net, gpus)` wraps net+renderer into a `DataParallel` module that splits along the **ray batch dim (`dim=1`)**. The "super-batch" `SB` is the object batch; `B` is rays per object.

### Config system (PyHocon)
`.conf` files with inheritance via `include required(...)`: `conf/default.conf` (single-view base) ← `conf/default_mv.conf` (multiview MLP) ← `conf/exp/<exp>.conf` (per-experiment overrides). Sections: `model`, `renderer`, `loss`, `train`. Factories (`make_model`, `make_encoder`, `make_mlp`, `*.from_conf`) construct objects from config subtrees, so adding a model/encoder/MLP variant means extending the relevant factory's type switch plus a `from_conf`.

### Datasets (kept in native formats, adapters built in)
`DVRDataset` handles `dvr` (ShapeNet NMR 64×64), `dvr_gen` (unseen-category split, `gen_` list prefix), and `dvr_dtu` (DTU; `new_` prefix, `sub_format=dtu`, custom z-bounds, color-jitter aug). `SRNDataset` = `srn` (single-category cars/chairs 128×128). `MultiObjectDataset` = `multi_obj` (two-object NeRF format). For SRN/multi_obj with `dir/cars_train`, `dir/cars_val`, pass `-D dir/cars`.

### Training loop
`train/trainlib/trainer.py::Trainer` is a generic base (data loaders, optimizer=Adam, checkpoint save/resume, train/eval/vis interval scheduling, TensorBoard). `train/train.py::PixelNeRFTrainer` subclasses it and implements `calc_losses` / `train_step` / `eval_step` / `vis_step`. Coarse+fine RGB losses (`model/loss.py`); bbox-guided ray sampling early in training (disabled after `--no_bbox_step`).

### Checkpoints (managed by `PixelNeRFNet.load_weights`/`save_weights`)
Under `checkpoints/<exp>/`: `pixel_nerf_latest` (model), `pixel_nerf_backup` (auto-backup before overwrite), `_renderer`, `_optim`, `_iter`, `_lrsched`. `--resume` reloads all of them. Drop a `pixel_nerf_init` to use as initialization. Pretrained weights extract to `checkpoints/`.
