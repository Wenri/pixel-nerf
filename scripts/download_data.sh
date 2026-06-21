#!/usr/bin/env bash
# Download pixelNeRF datasets / pretrained weights from this fork's GitHub Releases.
# These mirror the authors' original Google Drive data. No auth needed (public assets).
#
#   scripts/download_data.sh [--what data|weights|all] [--dest DIR] [--no-extract] [--no-verify]
#
# The SRN cars/chairs archives are repacked per split (cars_train/val/test, chairs_...),
# each a normal zip under GitHub's 2 GiB asset limit -- no reassembly needed.
set -euo pipefail

REPO_BASE="https://github.com/Wenri/pixel-nerf/releases/download"
DATA_TAG="data-v1"
WEIGHTS_TAG="weights-v1"

WHAT="all"
DEST=""
EXTRACT=1
VERIFY=1

usage() {
  cat <<'EOF'
Usage: scripts/download_data.sh [--what data|weights|all] [--dest DIR] [--no-extract] [--no-verify]

  --what        which release to fetch: data | weights | all   (default: all)
  --dest DIR    base target dir (default: ./data for datasets, ./checkpoints for weights)
  --no-extract  download (and verify) only; do not unzip
  --no-verify   skip the sha256sum -c integrity check
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --what) WHAT="${2:?}"; shift 2;;
    --dest) DEST="${2:?}"; shift 2;;
    --no-extract) EXTRACT=0; shift;;
    --no-verify) VERIFY=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 1;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 1; }; }
need curl; need unzip; need sha256sum

fetch() {  # $1=tag  $2=staging_dir  rest=assets
  local tag="$1" stage="$2"; shift 2
  mkdir -p "$stage"
  ( cd "$stage"
    for a in "$@"; do
      echo ">> $tag/$a"
      curl -fL --retry 5 --retry-delay 5 -O "$REPO_BASE/$tag/$a"
    done
    if [ "$VERIFY" -eq 1 ] && [ -f SHA256SUMS ]; then
      echo ">> verifying SHA256SUMS"
      sha256sum -c SHA256SUMS --ignore-missing
    fi
  )
}

unzip_each() {  # $1=target_dir  rest=zip globs/paths
  local target="$1"; shift
  mkdir -p "$target"
  local z
  for z in "$@"; do
    [ -e "$z" ] || continue
    unzip -qo "$z" -d "$target"
  done
}

do_data() {
  local base="${DEST:-./data}" stage
  stage="$base/.download"
  fetch "$DATA_TAG" "$stage" \
    chairs_train.zip chairs_val.zip chairs_test.zip \
    cars_train.zip cars_val.zip cars_test.zip \
    dtu_dataset.zip \
    multi_chair_train.zip multi_chair_val.zip multi_chair_test.zip \
    eval_out.zip SHA256SUMS
  if [ "$EXTRACT" -eq 1 ]; then
    unzip_each "$base/srn_chairs" "$stage"/chairs_train.zip "$stage"/chairs_val.zip "$stage"/chairs_test.zip
    unzip_each "$base/srn_cars"   "$stage"/cars_train.zip   "$stage"/cars_val.zip   "$stage"/cars_test.zip
    unzip_each "$base/dtu"        "$stage"/dtu_dataset.zip
    unzip_each "$base/multi_chair" "$stage"/multi_chair_train.zip "$stage"/multi_chair_val.zip "$stage"/multi_chair_test.zip
    unzip_each "$base"           "$stage"/eval_out.zip
    cat <<EOF

Datasets ready under $base/. Pass these -D paths:
  SRN chairs : -D $base/srn_chairs/chairs
  SRN cars   : -D $base/srn_cars/cars
  DTU        : -D $base/dtu/rs_dtu_4
  multi_chair: -D $base/multi_chair
(remove $stage to reclaim the downloaded zips)
EOF
  fi
}

do_weights() {
  local base="${DEST:-./checkpoints}" stage
  stage="$base/.download"
  fetch "$WEIGHTS_TAG" "$stage" \
    pixel_nerf_weights.zip multi_chair_1v_checkpoint.zip multi_chair_2v_checkpoint.zip SHA256SUMS
  if [ "$EXTRACT" -eq 1 ]; then
    unzip_each "$base" "$stage"/pixel_nerf_weights.zip "$stage"/multi_chair_1v_checkpoint.zip "$stage"/multi_chair_2v_checkpoint.zip
    echo ""
    echo "Weights extracted into $base/ (sn64, sn64_unseen, srn_car, srn_chair, dtu, ...)"
    echo "(remove $stage to reclaim the downloaded zips)"
  fi
}

case "$WHAT" in
  data) do_data;;
  weights) do_weights;;
  all) do_data; do_weights;;
  *) echo "invalid --what: $WHAT" >&2; usage; exit 1;;
esac
echo "Done."
