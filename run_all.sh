#!/usr/bin/env bash
# Run every lesson in order, then rebuild the site data. Figures go to outputs/.
# FULL=1 bash run_all.sh   runs the longer training versions.
set -e
cd "$(dirname "$0")"
for f in lessons/[0-9][0-9]_*.py; do
  echo; echo "################ $f ################"
  python "$f"
done
python site/build_data.py
