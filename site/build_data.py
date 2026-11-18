"""Collect every lesson's saved numbers into one file for the interactive page.

    python site/build_data.py        # reads outputs/results/*.json, writes site/data/results.js

The lessons themselves save their results (outputs/results/NN_name.json) when you
run them, because the training runs are too slow to repeat just for the page.
So: run the lessons first (bash run_all.sh), then this. Lessons you haven't run
yet show up on the page as "run lesson NN to fill this in".
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "outputs" / "results"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

data = {}
for f in sorted(RES.glob("*.json")):
    data[f.stem[:2]] = json.loads(f.read_text())
    print(f"  {f.name:32} {f.stat().st_size / 1024:7.1f} KB")

js = "window.RESULTS = " + json.dumps(data, separators=(",", ":")) + ";\n"
(OUT / "results.js").write_text(js)
print(f"wrote site/data/results.js ({len(js) / 1024:.0f} KB, lessons: {', '.join(sorted(data)) or 'none'})")
