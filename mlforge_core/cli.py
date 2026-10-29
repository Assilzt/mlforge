from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .trainer import predict, train_dataframe


def main() -> None:
    parser=argparse.ArgumentParser(prog="mlforge", description="Transparent tabular ML baseline trainer")
    sub=parser.add_subparsers(dest="command", required=True)
    t=sub.add_parser("train"); t.add_argument("data"); t.add_argument("--target", required=True); t.add_argument("--out", required=True); t.add_argument("--task", choices=["classification","regression"])
    p=sub.add_parser("predict"); p.add_argument("model"); p.add_argument("data"); p.add_argument("--out", required=True)
    args=parser.parse_args()
    if args.command=="train":
        path=Path(args.data); df=pd.read_excel(path) if path.suffix.lower() in {".xlsx",".xls"} else pd.read_csv(path); report=train_dataframe(df,args.target,args.out,args.task); print(f"Winner: {report['winner']} | metrics: {report['metrics']}")
    else: predict(args.model,args.data,args.out); print(f"Predictions written to {args.out}")
