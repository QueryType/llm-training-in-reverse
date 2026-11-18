#!/usr/bin/env bash
# Run every lesson in order, then rebuild the site data. Figures go to outputs/.
#   bash run_all.sh            ~1 hour on an M4 Pro (lesson 11's pretraining is ~12 min of it)
#   FULL=1 bash run_all.sh     longer training runs everywhere
set -e
cd "$(dirname "$0")"
for f in lessons/[0-9][0-9]_*.py; do
  echo; echo "################ $f ################"
  python "$f"
done
if python -c "import mlx.core" 2>/dev/null; then
  echo; echo "################ lessons/mlx_addendum.py ################"
  python lessons/mlx_addendum.py
fi
python site/build_data.py
