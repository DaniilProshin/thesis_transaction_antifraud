"""Хранилище IEEE-CIS. Каждая пара CSV образует неизменяемую версию набора."""

import csv
import hashlib
import io
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .core import CAT, FEATURE_VERSION, ID_CAT, ID_NUM, META, TX_CAT, Settings

DEFAULT_DB = os.environ.get("FRAUD_DB", "fraud.sqlite3")
ID_NUM_ALL = [f"id_{i:02}" for i in range(1, 12)]
ID_CAT_ALL = [f"id_{i:02}" for i in range(12, 39)] + ["DeviceType", "DeviceInfo"]
ID_ALL = ID_NUM_ALL + ID_CAT_ALL
FEATURES = META + TX_CAT + ID_CAT + ID_NUM + ["has_identity"]
VIEWS = ["acceptance", "metrics", "predictions", "queue", "reviews", "thresholds", "history"]


def dumps(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)


def columns(names):
    # Только константы схемы в коде. Значения из интерфейса всегда передаются отдельно.
    return ", ".join('"' + name + '"' for name in names)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sql_fields(names, sql_type):
    return ", ".join(f'"{name}" {sql_type}' for name in names)


DDL = f"""
CREATE TABLE IF NOT EXISTS dataset(
 id INTEGER PRIMARY KEY, role TEXT NOT NULL CHECK(role IN ('train','demo')),
 tx_sha TEXT NOT NULL, identity_sha TEXT NOT NULL, tx_rows INTEGER NOT NULL DEFAULT 0,
 id_rows INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(role, tx_sha, identity_sha));
CREATE TABLE IF NOT EXISTS txn(
 dataset_id INTEGER NOT NULL, TransactionID INTEGER NOT NULL,
 TransactionDT INTEGER NOT NULL CHECK(TransactionDT>=0), TransactionAmt REAL NOT NULL CHECK(TransactionAmt>=0),
 isFraud INTEGER CHECK(isFraud IN (0,1)), {sql_fields(TX_CAT, "TEXT")},
 PRIMARY KEY(dataset_id,TransactionID)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_time ON txn(dataset_id,TransactionDT,TransactionID);
CREATE TABLE IF NOT EXISTS identity(
 dataset_id INTEGER NOT NULL, TransactionID INTEGER NOT NULL,
 {sql_fields(ID_NUM_ALL, "REAL")}, {sql_fields(ID_CAT_ALL, "TEXT")},
 PRIMARY KEY(dataset_id,TransactionID),
 FOREIGN KEY(dataset_id,TransactionID) REFERENCES txn(dataset_id,TransactionID)) WITHOUT ROWID;
CREATE VIEW IF NOT EXISTS enriched AS
 SELECT t.*, i.TransactionID IS NOT NULL AS has_identity,
 {", ".join("i." + c for c in ID_ALL)}
 FROM txn t LEFT JOIN identity i USING(dataset_id,TransactionID);
CREATE TABLE IF NOT EXISTS profile(
 id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, params TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model(
 id INTEGER PRIMARY KEY, train_set INTEGER NOT NULL, profile_id INTEGER NOT NULL,
 created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, manifest TEXT NOT NULL, artifact BLOB NOT NULL, digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS operating_point(
 model_id INTEGER NOT NULL, variant TEXT NOT NULL, threshold REAL NOT NULL,
 validation TEXT NOT NULL, PRIMARY KEY(model_id,variant));
CREATE TABLE IF NOT EXISTS epoch(
 model_id INTEGER NOT NULL, epoch INTEGER NOT NULL, train_loss REAL NOT NULL,
 validation_loss REAL NOT NULL, PRIMARY KEY(model_id,epoch));
CREATE TABLE IF NOT EXISTS evaluation(
 id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS acceptance(
 evaluation_id INTEGER PRIMARY KEY, ap_gain REAL NOT NULL, recall REAL NOT NULL,
 fpr REAL NOT NULL, pass_gain INTEGER NOT NULL, pass_recall INTEGER NOT NULL, pass_fpr INTEGER NOT NULL,
 passed INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS evaluation_metric(
 evaluation_id INTEGER NOT NULL, variant TEXT NOT NULL, values_json TEXT NOT NULL,
 PRIMARY KEY(evaluation_id,variant));
CREATE TABLE IF NOT EXISTS evaluation_score(
 evaluation_id INTEGER NOT NULL, variant TEXT NOT NULL, TransactionID INTEGER NOT NULL,
 isFraud INTEGER NOT NULL CHECK(isFraud IN (0,1)), score REAL NOT NULL, is_alert INTEGER NOT NULL,
 PRIMARY KEY(evaluation_id,variant,TransactionID)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS demo_score(
 model_id INTEGER NOT NULL, dataset_id INTEGER NOT NULL, TransactionID INTEGER NOT NULL,
 score REAL NOT NULL CHECK(score BETWEEN 0 AND 1), residual REAL NOT NULL, oov INTEGER NOT NULL,
 is_alert INTEGER NOT NULL CHECK(is_alert IN(0,1)), PRIMARY KEY(model_id,dataset_id,TransactionID),
 FOREIGN KEY(dataset_id,TransactionID) REFERENCES txn(dataset_id,TransactionID)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_queue ON demo_score(model_id,dataset_id,score DESC,TransactionID);
CREATE TABLE IF NOT EXISTS review(
 id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL, dataset_id INTEGER NOT NULL, TransactionID INTEGER NOT NULL,
 decision TEXT NOT NULL CHECK(decision IN ('fraud','legit','uncertain')), comment TEXT NOT NULL,
 created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(model_id,dataset_id,TransactionID) REFERENCES demo_score(model_id,dataset_id,TransactionID));
CREATE INDEX IF NOT EXISTS ix_review ON review(model_id,dataset_id,TransactionID,id DESC);
"""


@contextmanager
def connect(path):
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:
            yield conn
    finally:
        conn.close()


def initialize(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(DDL)
        db.execute(
            "INSERT OR IGNORE INTO profile(name,params) VALUES(?,?)",
            ("default", dumps(asdict(Settings())))
        )


def records(path, sql, args=()):
    with connect(path) as db:
        return [dict(r) for r in db.execute(sql, args)]


def required(path, sql, args):
    result = records(path, sql, args)
    if not result:
        raise ValueError("Запись не найдена. Проверьте ID и тип набора.")
    return result[0]


def create_profile(path, name, updates):
    cfg = Settings(**updates)
    if not name.strip():
        raise ValueError("Укажите имя профиля")
    with connect(path) as db:
        db.execute("INSERT INTO profile(name,params) VALUES(?,?)", (name, dumps(asdict(cfg))))


def dataset(path, role, identifier=None):
    if identifier is None:
        return required(
            path, "SELECT * FROM dataset WHERE role=? ORDER BY id DESC LIMIT 1", (role,)
        )
    return required(path, "SELECT * FROM dataset WHERE role=? AND id=?", (role, identifier))


def _headers(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    canonical = [re.sub(r"^id-(\d{2})$", r"id_\1", k) for k in header]
    if len(set(canonical)) != len(header):
        raise ValueError(f"{path.name}: повтор имени столбца после нормализации")
    return dict(zip(header, canonical))


def _read_chunks(path, identity, labeled):
    names = _headers(path)
    target = (
        ["TransactionID", *ID_ALL] if identity else META + TX_CAT + (["isFraud"] if labeled else [])
    )
    mandatory = ["TransactionID", *ID_CAT, *ID_NUM] if identity else target
    missing = set(mandatory) - set(names.values())
    if missing:
        raise ValueError(f"{path.name}: отсутствуют поля {sorted(missing)}")
    originals = [k for k, v in names.items() if v in target]
    for frame in pd.read_csv(
        path,
        usecols=originals,
        dtype="string",
        encoding="utf-8-sig",
        keep_default_na=False,
        na_values=[""],
        chunksize=10000,
    ):
        frame = frame.rename(columns=names).reindex(columns=target)
        integers = ["TransactionID"] if identity else ["TransactionID", "TransactionDT"]
        for name in integers:
            s = frame[name]
            if not s.str.fullmatch(r"[0-9]+").fillna(False).all():
                raise ValueError(f"{path.name}: {name} должен быть неотрицательным целым")
            frame[name] = s.astype("int64")
            if (frame[name] < 0).any():
                raise ValueError(f"{name} выходит за диапазон SQLite INTEGER")
        for name in ID_NUM_ALL if identity else ["TransactionAmt"]:
            arr = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(arr).any() or (not identity and (np.isnan(arr).any() or (arr < 0).any())):
                raise ValueError(f"{path.name}: недопустимое число в {name}")
            frame[name] = arr
        if not identity and labeled:
            if not frame.isFraud.isin(["0", "1"]).all():
                raise ValueError(f"{path.name}: isFraud должен быть 0 или 1")
            frame["isFraud"] = frame.isFraud.astype("int64")
        for name in (
            ["card1", "card2", "card3", "card5", "addr1", "addr2"]
            if not identity
            else [k for k in ID_CAT_ALL if k.startswith("id_")]
        ):
            frame[name] = (
                frame[name].astype("string").str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
            )
        # sqlite3 принимает Python int/float/None. pd.NA и numpy.int64 не привязываем.
        yield frame.astype(object).where(frame.notna(), None)


def import_pair(db_path, directory, role):
    """Обе таблицы и метаданные записываются атомарно. Ошибка оставляет старые наборы доступными."""
    if role not in {"train", "demo"}:
        raise ValueError("role: train или demo")
    prefix = "train" if role == "train" else "test"
    paths = [Path(directory) / f"{prefix}_{part}.csv" for part in ("transaction", "identity")]
    stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    hashes = [file_hash(p) for p in paths]
    with connect(db_path) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT id FROM dataset WHERE role=? AND tx_sha=? AND identity_sha=?", (role, *hashes)
        ).fetchone()
        if row:
            return row[0]
        ds = db.execute(
            "INSERT INTO dataset(role,tx_sha,identity_sha) VALUES(?,?,?)", (role, *hashes)
        ).lastrowid
        counts = []
        for table, path in zip(("txn", "identity"), paths):
            count = 0
            for frame in _read_chunks(path, table == "identity", role == "train"):
                names = ["dataset_id", *frame.columns]
                sql = (
                    f"INSERT INTO {table} ({columns(names)}) VALUES({','.join('?' for _ in names)})"
                )
                db.executemany(sql, ((ds, *r) for r in frame.itertuples(index=False, name=None)))
                count += len(frame)
            counts.append(count)
        if not counts[0]:
            raise ValueError(f"{paths[0].name}: файл транзакций пуст")
        if stamps != [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]:
            raise ValueError("Исходный файл изменился во время импорта. Повторите импорт.")
        db.execute("UPDATE dataset SET tx_rows=?,id_rows=? WHERE id=?", (*counts, ds))
        return ds


def blocks(path, ds, *, low=None, high=None, rows=4096, unscored_model=None, limit=None):
    """Единое SQL-обогащение train/demo. На один ключ максимум одна identity."""
    predicates, args = ["e.dataset_id=?"], [ds]
    for op, bound in ((">=", low), ("<", high)):
        if bound is not None:
            predicates.append(f"e.TransactionDT {op} ?")
            args.append(int(bound))
    if unscored_model is not None:
        predicates.append("""NOT EXISTS (SELECT 1 FROM demo_score s WHERE s.model_id=?
            AND s.dataset_id=e.dataset_id AND s.TransactionID=e.TransactionID)""")
        args.append(unscored_model)
    sql = (
        f"SELECT {columns(FEATURES + ['isFraud'])} FROM enriched e WHERE "
        + " AND ".join(predicates)
        + " ORDER BY e.TransactionDT,e.TransactionID"
    )
    if limit is not None:
        sql += " LIMIT ?"
        args.append(int(limit))
    with connect(path) as db:
        for frame in pd.read_sql_query(sql, db, params=args, chunksize=rows):
            if not frame.empty:
                frame[ID_NUM] = frame[ID_NUM].apply(pd.to_numeric)
                for name in CAT:
                    frame[name] = frame[name].astype("category")
                yield frame


def time_at(path, ds, offset):
    return required(
        path,
        """SELECT TransactionDT AS t FROM txn WHERE dataset_id=?
                    ORDER BY TransactionDT,TransactionID LIMIT 1 OFFSET ?""",
        (ds, int(offset)),
    )["t"]


def save_model(path, ds, profile_id, model, manifest, calibration, history):
    out = io.BytesIO()
    joblib.dump(model, out, compress=3)
    blob = out.getvalue()
    with connect(path) as db:
        mid = db.execute(
            """INSERT INTO model(train_set,profile_id,manifest,artifact,digest)
                            VALUES(?,?,?,?,?)""",
            (ds, profile_id, dumps(manifest), blob, hashlib.sha256(blob).hexdigest()),
        ).lastrowid
        db.executemany(
            "INSERT INTO operating_point VALUES(?,?,?,?)",
            [(mid, v, t, dumps(m)) for v, (t, m) in calibration.items()],
        )
        db.executemany(
            "INSERT INTO epoch VALUES(?,?,?,?)",
            [(mid, h["epoch"], h["train_loss"], h["validation_loss"]) for h in history],
        )
    return mid


def load_model(path, mid):
    row = required(path, "SELECT * FROM model WHERE id=?", (mid,))
    blob = row.pop("artifact")
    if hashlib.sha256(blob).hexdigest() != row["digest"]:
        raise ValueError("Контрольная сумма модели не совпадает")
    manifest = json.loads(row["manifest"])
    if (
        manifest["feature_version"] != FEATURE_VERSION
        or manifest["categorical"] != CAT
        or manifest["identity_numeric"] != ID_NUM
    ):
        raise ValueError("Для этой модели требуется исходная версия схемы признаков")
    # joblib загружается только из локальной доверенной БД, созданной нашим CLI.
    detector = joblib.load(io.BytesIO(blob))
    thresholds = records(
        path, "SELECT variant,threshold FROM operating_point WHERE model_id=?", (mid,)
    )
    return (
        detector,
        row["train_set"],
        manifest,
        {r["variant"]: r["threshold"] for r in thresholds},
    )


def save_review(path, mid, ds, tid, decision, comment):
    if len(comment) > 2000:
        raise ValueError("Комментарий длиннее 2000 символов")
    with connect(path) as db:
        db.execute(
            """INSERT INTO review(model_id,dataset_id,TransactionID,decision,comment)
                      VALUES(?,?,?,?,?)""",
            (mid, ds, tid, decision, comment),
        )


def report_query(kind, identifier, demo=None, alerts=False):
    """Единый запрос для просмотра и двух способов экспорта."""
    if kind == "acceptance":
        return ("SELECT * FROM acceptance WHERE evaluation_id=?", (identifier,))
    if kind == "metrics":
        return (
            "SELECT variant,values_json FROM evaluation_metric WHERE evaluation_id=? ORDER BY variant",
            (identifier,),
        )
    if kind == "predictions":
        return (
            """SELECT * FROM evaluation_score WHERE evaluation_id=?
                   ORDER BY variant,TransactionID""",
            (identifier,),
        )
    if kind == "thresholds":
        return ("SELECT * FROM operating_point WHERE model_id=? ORDER BY variant", (identifier,))
    if kind == "history":
        return ("SELECT * FROM epoch WHERE model_id=? ORDER BY epoch", (identifier,))
    if kind == "reviews":
        return (
            "SELECT * FROM review WHERE model_id=? AND dataset_id=? ORDER BY id",
            (identifier, demo),
        )
    if kind != "queue":
        raise ValueError("Неизвестный отчет")
    sql = """SELECT s.TransactionID,t.TransactionDT,t.TransactionAmt,t.ProductCD,
        t.has_identity,t.DeviceType,t.DeviceInfo,s.score,s.residual,s.oov,s.is_alert,
        (SELECT r.decision FROM review r WHERE r.model_id=s.model_id AND r.dataset_id=s.dataset_id
         AND r.TransactionID=s.TransactionID ORDER BY r.id DESC LIMIT 1) AS last_decision
        FROM demo_score s JOIN enriched t USING(dataset_id,TransactionID)
        WHERE s.model_id=? AND s.dataset_id=?"""
    if alerts:
        sql += " AND s.is_alert=1"
    return sql + " ORDER BY s.score DESC,s.TransactionID", (identifier, demo)


def export_csv(path, kind, identifier, stream, demo=None, alerts=False):
    sql, args = report_query(kind, identifier, demo, alerts)
    with connect(path) as db:
        cur = db.execute(sql, args)
        writer = csv.writer(stream)
        if kind == "metrics":
            expanded = [{"variant": r["variant"], **json.loads(r["values_json"])} for r in cur]
            if expanded:
                names = list(expanded[0])
                writer.writerow(names)
                writer.writerows([[row[k] for k in names] for row in expanded])
            return
        writer.writerow([field[0] for field in cur.description])
        writer.writerows(cur)
