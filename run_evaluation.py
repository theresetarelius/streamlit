import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gt",     required=True)
parser.add_argument("--result", required=True)
parser.add_argument("--out",    default="evaluation_output")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

print(f"Laddar {args.result}...")
result = np.load(args.result, allow_pickle=True).item()

print(f"Kör utvärdering mot {args.gt}...")
from reactiv_evaluation import evaluate_reactiv
save_path = os.path.join(args.out, "overlay.png")
results = evaluate_reactiv(
    reactiv_output=result,
    gt_path=args.gt,
    save_path=save_path
)
print("Resultat:", results)
print(f"Kollar om fil finns: {os.path.exists(save_path)}")
print("Klar!")

"""venv/bin/python -u run_evaluation.py \
    --gt /opt/saab/mex/streamlit/GT/la_2.tif \
    --result /opt/saab/mex/streamlit/results/reactiv_result.npy \
    --out evaluation_output 2>&1 """