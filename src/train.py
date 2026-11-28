import argparse
import json
import math
import sqlite3
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from . import core
from . import storage as db


def labeled_frame(path, ds, **interval):
    pieces = list(db.blocks(path, ds, rows=20000, **interval))
    if not pieces:
        raise ValueError("Пустой временной интервал")
    
    frame = pd.concat(pieces, ignore_index=True)
    for name in core.CAT:
        frame[name] = frame[name].astype("category")
    
    return frame


def choose_threshold(y, scores, cfg, variant):
    if cfg.threshold_mode == "fixed":
        return float(cfg.fixed_ae_threshold if variant == "AE_only" else cfg.fixed_threshold)
    
    fpr, recall, tau = roc_curve(y, scores, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= cfg.max_fpr)
    selected = tau[valid[np.argmax(recall[valid])]]

    return float(selected if np.isfinite(selected) else np.nextafter(float(scores.max()), np.inf))


def metrics(y, score, threshold, ids):
    predicted = score >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    top = np.lexsort((ids, -score))[: max(1, math.ceil(len(y) / 100))]
    precision, recall = tp / max(1, tp + fp), tp / (tp + fn)
    
    return {
        "rows": len(y),
        "positives": tp + fn,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "AP": float(average_precision_score(y, score)),
        "ROC_AUC": float(roc_auc_score(y, score)),
        "precision": precision,
        "recall": recall,
        "FPR": fp / (tn + fp),
        "F1": (2 * tp / max(1, 2 * tp + fp + fn)),
        "alerts": int(predicted.sum()),
        "recall_top_1pct": float(y[top].sum() / y.sum()),
        "threshold": threshold,
    }


def collect_scores(path, ds, model, variants, **interval):
    observed, identifiers = [], []
    collected = {v: [] for v in variants}
    for frame in db.blocks(path, ds, **interval):
        batch, _, _ = model.predict(frame, variants)
        observed.append(frame.isFraud.to_numpy(dtype=np.int8))
        identifiers.append(frame.TransactionID.to_numpy())
        for v in variants:
            collected[v].append(batch[v].astype(np.float64))
    
    if not observed or len(np.unique(np.concatenate(observed))) != 2:
        raise ValueError("Для расчета метрик во временном интервале нужны оба класса")
    
    return (
        np.concatenate(observed),
        np.concatenate(identifiers),
        {v: np.concatenate(s) for v, s in collected.items()},
    )


def train(path, train_set=None, profile="default"):
    data = db.dataset(path, "train", train_set)
    ds, n = data["id"], data["tx_rows"]
    setting = db.required(path, "SELECT id,params FROM profile WHERE name=?", (profile,))
    cfg = core.Settings(**json.loads(setting["params"]))
    a, b, v = [db.time_at(path, ds, int(n * part)) for part in (cfg.end_a, cfg.end_b, cfg.end_v)]
    
    counts = db.required(
        path,
        """SELECT sum(TransactionDT < ?) AS A,
        sum(TransactionDT >= ? AND TransactionDT < ?) AS B,
        sum(TransactionDT >= ? AND TransactionDT < ?) AS V, sum(TransactionDT >= ?) AS T
        FROM txn WHERE dataset_id=?""",
        (a, a, b, b, v, v, ds),
    )
    if min(counts.values()) < 2 or counts["A"] + counts["B"] >= 500000:
        raise ValueError(f"Некорректные временные интервалы или превышен лимит обучения: {counts}")
    
    stop_cut = db.time_at(path, ds, int(counts["A"] * cfg.ae_fit_share))
    if stop_cut >= a:
        raise ValueError("Недостаточно временных отметок для ранней остановки")
    
    fit = labeled_frame(path, ds, high=stop_cut)
    stop = labeled_frame(path, ds, low=stop_cut, high=a)
    prep = core.Preprocessor().fit(fit, cfg)
    net, history = core.train_ae(prep, fit, stop, cfg)
    del fit, stop
    train_b = labeled_frame(path, ds, low=a, high=b)
    y = train_b.isFraud.to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Для обучения RF на блоке B нужны оба класса")
    
    raw, emb, errors, _ = core.features(prep, net, train_b)
    forests = {}
    variants = ["Hybrid", "RF_base"]
    if cfg.extended:
        variants += ["RF_E", "RF_R"]
    
    for variant in variants:
        forests[variant] = core.new_forest(cfg).fit(core.forest_input(raw, emb, errors, variant), y)
    
    del raw, emb, errors, train_b
    if cfg.extended:
        from scipy import sparse

        raw_parts, labels = [], []
        for frame in db.blocks(path, ds, high=b):
            raw_parts.append(prep.raw(*prep.arrays(frame)))
            labels.append(frame.isFraud.to_numpy())
        forests["RF_full"] = core.new_forest(cfg).fit(
            sparse.vstack(raw_parts), np.concatenate(labels)
        )
        del raw_parts, labels

    for forest in forests.values():
        forest.n_jobs = 1
    
    detector = core.Detector(prep, net, forests)
    yv, ids, scores = collect_scores(path, ds, detector, [*forests, "AE_only"], low=b, high=v)
    calibration = {}
    for name, score in scores.items():
        tau = choose_threshold(yv, score, cfg, name)
        calibration[name] = tau, metrics(yv, score, tau, ids)
    
    manifest = {
        "feature_version": core.FEATURE_VERSION,
        "categorical": core.CAT,
        "identity_numeric": core.ID_NUM,
        "numeric_input_dim": prep.n_num,
        "settings": asdict(cfg),
        "cuts": {"early_stop": stop_cut, "B_start": a, "V_start": b, "T_start": v},
        "rows": counts,
        "method": {
            "rf_class_weight": "balanced_subsample",
            "rf_max_features": "sqrt",
            "rf_predict_threads": 1,
            "optimizer": "Adam",
            "clip_grad_norm": 5.0,
            "min_stop_improvement": 1e-5,
            "embedding_dim_rule": "min(16,max(2,ceil(sqrt(K))))",
            "threshold_tie_break": "highest threshold at maximal recall",
            "train_rows_limit_exclusive": 500000,
        },
        "train_tx_sha": data["tx_sha"],
        "train_identity_sha": data["identity_sha"],
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "joblib": joblib.__version__,
        },
    }
    return db.save_model(path, ds, setting["id"], detector, manifest, calibration, history)


def evaluate(path, mid):
    detector, ds, manifest, thresholds = db.load_model(path, mid)
    torch.set_num_threads(detector.net.cfg.threads)
    y, ids, scores = collect_scores(path, ds, detector, thresholds, low=manifest["cuts"]["T_start"])
    with db.connect(path) as conn:
        eid = conn.execute("INSERT INTO evaluation(model_id) VALUES(?)", (mid,)).lastrowid
        results = {}
        for variant, score in scores.items():
            tau = thresholds[variant]
            values = metrics(y, score, tau, ids)
            results[variant] = values
            conn.execute(
                "INSERT INTO evaluation_metric VALUES(?,?,?)", (eid, variant, db.dumps(values))
            )
            conn.executemany(
                "INSERT INTO evaluation_score VALUES(?,?,?,?,?,?)",
                (
                    (eid, variant, int(i), int(target), float(s), int(s >= tau))
                    for i, target, s in zip(ids, y, score)
                ),
            )
        hybrid = results["Hybrid"]
        cfg = detector.net.cfg
        gain = hybrid["AP"] - results["RF_base"]["AP"]
        checks = [
            gain >= cfg.ap_gain_min,
            hybrid["recall"] >= cfg.test_recall_min,
            hybrid["FPR"] <= cfg.test_fpr_max,
        ]
        conn.execute(
            "INSERT INTO acceptance VALUES(?,?,?,?,?,?,?,?)",
            (eid, gain, hybrid["recall"], hybrid["FPR"], *checks, all(checks)),
        )
    return eid


def score_demo(path, mid, demo_set=None, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("Размер пакета должен быть положительным")
    ds = db.dataset(path, "demo", demo_set)["id"]
    model, _, _, thresholds = db.load_model(path, mid)
    torch.set_num_threads(model.net.cfg.threads)
    tau, inserted = thresholds["Hybrid"], 0
    for frame in db.blocks(path, ds, unscored_model=mid, limit=limit):
        batch, residual, oov = model.predict(frame, ["Hybrid"])
        rows = (
            (mid, ds, int(tid), float(s), float(r), int(k), int(s >= tau))
            for tid, s, r, k in zip(frame.TransactionID, batch["Hybrid"], residual, oov)
        )
        with db.connect(path) as conn:
            conn.executemany(
                """INSERT INTO demo_score VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(model_id,dataset_id,TransactionID) DO NOTHING""",
                rows,
            )
            inserted += conn.total_changes
    return inserted


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default=db.DEFAULT_DB, help="База SQLite (также переменная FRAUD_DB)"
    )
    cmds = parser.add_subparsers(dest="cmd", required=True)
    cmds.add_parser("init", help="Создать пустую БД и профиль default")
    imp = cmds.add_parser("import", help="Импортировать версии пар CSV")
    imp.add_argument("--data", type=Path, default=Path("data"))
    imp.add_argument("--role", choices=["train", "demo", "all"], default="all")
    config = cmds.add_parser(
        "config", help="Создать именованный профиль из JSON, остальные параметры по умолчанию"
    )
    config.add_argument("--name", required=True)
    config.add_argument(
        "--params", required=True, help='Например: {"max_fpr":0.005,"extended":true}'
    )
    tr = cmds.add_parser(
        "train", help="Обучить, выбрать пороги на V и автоматически проверить на T"
    )
    tr.add_argument("--train-set", type=int)
    tr.add_argument("--profile", default="default")
    ev = cmds.add_parser(
        "evaluate", help="Повторная автоматическая проверка сохраненной модели на T"
    )
    ev.add_argument("--model", type=int, required=True)
    for command in (tr, ev):
        command.add_argument(
            "--require-pass",
            action="store_true",
            help="После записи отчета вернуть код 3 при недостижении критериев"
        )
    score = cmds.add_parser(
        "score-demo", help="Рассчитать недостающие оценки для демонстрационного набора"
    )
    score.add_argument("--model", type=int, required=True)
    score.add_argument("--demo", type=int)
    score.add_argument("--limit", type=int)
    cmds.add_parser("list", help="Показать наборы, профили, модели и отчеты")
    export = cmds.add_parser("export", help="Выгрузить отчет в CSV по явному запросу")
    export.add_argument("kind", choices=db.VIEWS)
    group = export.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=int)
    group.add_argument("--evaluation", type=int)
    export.add_argument("--demo", type=int)
    export.add_argument("--alerts", action="store_true")
    export.add_argument("--out", type=Path, help="По умолчанию CSV направляется в stdout")
    args = parser.parse_args()

    path = args.db
    db.initialize(path)
    if args.cmd == "init":
        return {"database": path}
    if args.cmd == "import":
        roles = ("train", "demo") if args.role == "all" else (args.role,)
        return {role: db.import_pair(path, args.data, role) for role in roles}
    if args.cmd == "config":
        params = json.loads(args.params)
        if not isinstance(params, dict):
            raise ValueError("--params должен содержать объект JSON")
        db.create_profile(path, args.name, params)
        return {"profile": args.name}
    if args.cmd in {"train", "evaluate"}:
        if args.cmd == "train":
            mid = train(path, args.train_set, args.profile)
            print(f"Модель {mid} сохранена. Проверка на T…", file=sys.stderr)
        else:
            mid = args.model
        eid = evaluate(path, mid)
        check = db.required(path, "SELECT * FROM acceptance WHERE evaluation_id=?", (eid,))
        result = {"model": mid, "evaluation": eid, "acceptance": check}
        if args.require_pass and not check["passed"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(3)
        return result
    if args.cmd == "score-demo":
        return {"inserted": score_demo(path, args.model, args.demo, args.limit)}
    if args.cmd == "list":
        return {
            "datasets": db.records(path, "SELECT * FROM dataset ORDER BY id"),
            "profiles": db.records(path, "SELECT * FROM profile ORDER BY id"),
            "models": db.records(
                path, "SELECT id,train_set,profile_id,created FROM model ORDER BY id"
            ),
            "evaluations": db.records(path, "SELECT * FROM evaluation ORDER BY id")
        }
    eval_kind = args.kind in {"acceptance", "metrics", "predictions"}
    if eval_kind != (args.evaluation is not None):
        parser.error(
            "Для acceptance/metrics/predictions нужен --evaluation, для остальных отчетов --model"
        )
    if args.alerts and args.kind != "queue":
        parser.error("--alerts используется только для queue")
    identifier = args.evaluation if eval_kind else args.model
    table = "evaluation" if eval_kind else "model"
    db.required(path, f"SELECT id FROM {table} WHERE id=?", (identifier))
    ds = None
    if args.kind in {"queue", "reviews"}:
        ds = db.dataset(path, "demo", args.demo)["id"]
    elif args.demo is not None:
        parser.error("--demo используется только для queue/reviews")
    if args.out is not None and args.out.resolve() == Path(path).resolve():
        parser.error("Экспорт в файл базы данных запрещен")

    dest = (
        args.out.open("x", encoding="utf-8-sig", newline="")
        if args.out
        else nullcontext(sys.stdout)
    )
    with dest as stream:
        db.export_csv(path, args.kind, identifier, stream, ds, args.alerts)


if __name__ == "__main__":
    try:
        result = cli()
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"Ошибка: {exc}") from exc
