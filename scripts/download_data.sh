#!/usr/bin/env bash
# Downloads the datasets required by the study (specification Step 2).
#
#   ./scripts/download_data.sh                # primary dataset only
#   ./scripts/download_data.sh --external     # also the Figshare external set
#
# Requires Kaggle API credentials: either ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and
# KAGGLE_KEY in the project's .env file (see .env.example).

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
INCLUDE_EXTERNAL=0
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --external) INCLUDE_EXTERNAL=1 ;;
    --force)    FORCE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Primary dataset: four classes, the main training / validation / internal test source.
PRIMARY_SLUG="mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images"
PRIMARY_DIR="${DATA_DIR}/raw/bt_mri"

# External dataset A: three classes, used for cross-dataset validation in Step 17.
EXTERNAL_SLUG="ashkhagan/figshare-brain-tumor-dataset"
EXTERNAL_DIR="${DATA_DIR}/raw/figshare"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

command -v kaggle >/dev/null 2>&1 || {
  echo "The 'kaggle' CLI was not found. Install it with: pip install kaggle" >&2
  exit 1
}

fetch() {
  local slug="$1" dest="$2"

  if [ "$FORCE" -eq 0 ] && [ -d "$dest" ] && [ -n "$(find "$dest" -type f -print -quit 2>/dev/null)" ]; then
    echo "Already present: $dest ($(find "$dest" -type f | wc -l) files)"
    return
  fi

  echo "Downloading $slug -> $dest"
  mkdir -p "$dest"
  kaggle datasets download -d "$slug" -p "$dest" --unzip
  echo "Done: $(find "$dest" -type f | wc -l) files in $dest"
}

fetch "$PRIMARY_SLUG" "$PRIMARY_DIR"

# The archive nests its content one or two levels deep and the exact layout varies by
# mirror, so report what actually landed rather than assuming.
echo
echo "Class folders discovered under ${PRIMARY_DIR}:"
find "$PRIMARY_DIR" -maxdepth 3 -type d | while read -r dir; do
  n=$(find "$dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  [ "$n" -gt 0 ] && printf '  %-60s %6s images\n' "$dir" "$n"
done

if [ "$INCLUDE_EXTERNAL" -eq 1 ]; then
  fetch "$EXTERNAL_SLUG" "$EXTERNAL_DIR"
fi

cat <<'EOF'

If the class folders sit deeper than data/raw/bt_mri, point the datamodule at them:
  python src/analyze.py analysis=step04_audit data.raw_subdir=raw/bt_mri/<subfolder>

Next: python src/analyze.py analysis=step04_audit
EOF
