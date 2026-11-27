import math
from copy import deepcopy
from dataclasses import dataclass, fields

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

TX_CAT = [
    "ProductCD",
    *[f"card{i}" for i in range(1, 7)],
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    *[f"M{i}" for i in range(1, 10)],
]
ID_CAT = ["DeviceType", "DeviceInfo", "id_12", "id_15", "id_28", "id_29", "id_30", "id_31", "id_33"]
ID_NUM = ["id_01", "id_02", "id_05"]
CAT = TX_CAT + ID_CAT
META = ["TransactionID", "TransactionDT", "TransactionAmt"]
FEATURE_VERSION = 1


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    epochs: int = 20
    patience: int = 3
    batch_size: int = 1024
    min_count: int = 20
    max_categories: int = 512
    hidden: int = 64
    latent: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    mask_rate: float = 0.05
    noise: float = 0.01
    trees: int = 200
    depth: int = 14
    min_leaf: int = 10
    threads: int = 2
    max_fpr: float = 0.01
    threshold_mode: str = "validation_fpr"
    fixed_threshold: float = 0.5
    fixed_ae_threshold: float = 1.0  # AE_only ранжирует по ошибке, у нее своя шкала.
    extended: bool = False
    # Границы долей кумулятивные: конец A, конец B, конец V. Последним остается T.
    end_a: float = 0.40
    end_b: float = 0.70
    end_v: float = 0.85
    ae_fit_share: float = 0.90
    test_recall_min: float = 0.60
    test_fpr_max: float = 0.015
    ap_gain_min: float = 0.01

    def __post_init__(self):
        for field in fields(self):
            x = getattr(self, field.name)
            if field.type is int and (type(x) is not int or x < (0 if field.name == "seed" else 1)):
                raise ValueError(f"{field.name}: требуется целое допустимое значение")
            if field.type is float and (type(x) not in (float, int) or not math.isfinite(x)):
                raise ValueError(f"{field.name}: требуется конечное число")
        
        if not 0 <= self.seed < 2**32 or type(self.extended) is not bool:
            raise ValueError("Некорректный seed или extended")
        
        if self.threshold_mode not in {"validation_fpr", "fixed"}:
            raise ValueError("threshold_mode: validation_fpr или fixed")
        
        if not all(0 <= x <= 1 for x in (self.max_fpr, self.fixed_threshold, self.mask_rate)):
            raise ValueError("Пороги и mask_rate должны быть в [0, 1]")
        
        if not 0 < self.end_a < self.end_b < self.end_v < 1 or not 0 < self.ae_fit_share < 1:
            raise ValueError("Границы временных блоков должны строго возрастать внутри (0, 1)")
        
        if not all(
            0 <= x <= 1 for x in (self.test_recall_min, self.test_fpr_max, self.ap_gain_min)
        ):
            raise ValueError("Целевые показатели проверки должны принадлежать [0, 1]")
        
        if (
            self.learning_rate <= 0
            or min(self.weight_decay, self.noise, self.fixed_ae_threshold) < 0
        ):
            raise ValueError("Некорректные параметры оптимизации")


class Preprocessor:
    def fit(self, frame, cfg):
        self.maps, self.seen, self.sizes = [], [], []

        for name in CAT:
            counts = frame[name].dropna().astype(str).value_counts()
            
            frequent = sorted(
                ((v, n) for v, n in counts.items() if n >= cfg.min_count),
                key=lambda pair: (-pair[1], pair[0]),
            )[: cfg.max_categories]

            self.maps.append({v: i + 3 for i, (v, _) in enumerate(frequent)})
            self.seen.append(set(counts.index))
            self.sizes.append(3 + len(frequent))
        
        self.medians = frame[ID_NUM].median().fillna(0)
        
        observed = frame[ID_NUM]
        iqr = observed.quantile(0.75) - observed.quantile(0.25)
        fallback = observed.std(ddof=0)
        self.id_scale = iqr.where(iqr > 0, fallback).fillna(1).replace(0, 1)
        self.scaler = RobustScaler().fit(self.payment_numbers(frame))
        self.n_num = 3 + 2 * len(ID_NUM) + 1
        self.onehot = OneHotEncoder(
            categories=[np.arange(k) for k in self.sizes], dtype=np.float32, sparse_output=True
        )
        self.onehot.fit(np.zeros((1, len(CAT)), dtype=np.int64))
        return self

    @staticmethod
    def payment_numbers(frame):
        phase = 2 * np.pi * (frame.TransactionDT.to_numpy() % 86400) / 86400
        return np.column_stack((np.log1p(frame.TransactionAmt), np.sin(phase), np.cos(phase)))

    def arrays(self, frame):
        observed = frame[ID_NUM]
        scaled = (observed.fillna(self.medians) - self.medians) / self.id_scale

        nums = np.column_stack(
            (
                self.scaler.transform(self.payment_numbers(frame)),
                scaled,
                observed.isna(),
                frame.has_identity,
            )
        ).astype("float32")

        codes = np.empty((len(frame), len(CAT)), dtype="int64")

        for j, name in enumerate(CAT):
            s = frame[name].astype("string")
            v = s.map(self.maps[j]).fillna(0).to_numpy(dtype="int64", copy=True)
            v[(v == 0) & s.isin(self.seen[j]).to_numpy()] = 2
            v[s.isna().to_numpy()] = 1  # 0 — новое, 1 — пропуск, 2 — известное редкое.
            codes[:, j] = v
        
        return nums, codes

    def raw(self, nums, codes):
        return sparse.hstack((sparse.csr_matrix(nums), self.onehot.transform(codes)), format="csr")


class MixedAE(nn.Module):
    def __init__(self, prep, cfg):
        super().__init__()
        dims = [min(16, max(2, math.ceil(math.sqrt(n)))) for n in prep.sizes]
        self.emb = nn.ModuleList(
            [nn.Embedding(n, d, padding_idx=0) for n, d in zip(prep.sizes, dims)]
        )
        self.encoder = nn.Sequential(
            nn.Linear(prep.n_num + sum(dims), cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(nn.Linear(cfg.latent, cfg.hidden), nn.ReLU())
        self.num_head = nn.Linear(cfg.hidden, prep.n_num)
        self.cat_heads = nn.ModuleList([nn.Linear(cfg.hidden, k) for k in prep.sizes])
        self.cfg = cfg

    def forward(self, n, c):
        ni, ci = n, c

        if self.training:
            ni = n + self.cfg.noise * torch.randn_like(n)
            ci = c.masked_fill(torch.rand(c.shape, device=c.device) < self.cfg.mask_rate, 0)
        
        e = torch.cat([layer(ci[:, j]) for j, layer in enumerate(self.emb)], dim=1)
        h = self.decoder(self.encoder(torch.cat((ni, e), dim=1)))
        squared = (self.num_head(h) - n).square()
        use = torch.ones_like(squared)
        k = len(ID_NUM)

        use[:, 3 : 3 + k] = 1 - n[:, 3 + k : 3 + 2 * k]
        num_loss = (squared * use).sum(1) / use.sum(1)

        losses = [
            F.cross_entropy(head(h), c[:, j], reduction="none") / math.log(head.out_features)
            for j, head in enumerate(self.cat_heads)
        ]
        known = c.ne(0)
        cat_loss = (torch.stack(losses, dim=1) * known).sum(1) / known.sum(1).clamp_min(1)
        return e, num_loss + cat_loss


def train_ae(prep, fit, stop, cfg):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    net = MixedAE(prep, cfg)
    loaders = []

    for frame, shuffle in [(fit, True), (stop, False)]:
        normal = frame.loc[frame.isFraud == 0]
        if normal.empty:
            raise ValueError("Для обучения и ранней остановки AE нужны легитимные операции")
        arrays = prep.arrays(normal)
        data = TensorDataset(*(torch.from_numpy(a) for a in arrays))
        loaders.append(DataLoader(data, batch_size=cfg.batch_size, shuffle=shuffle))
    
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best, best_weights, idle, history = math.inf, None, 0, []
    
    for epoch in range(1, cfg.epochs + 1):
        means = []
        for loader, training in zip(loaders, [True, False]):
            net.train(training)
            total = 0.0
            with torch.set_grad_enabled(training):
                for n, c in loader:
                    _, errors = net(n, c)
                    loss = errors.mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Нечисловая ошибка обучения AE")
                    if training:
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                        opt.step()
                    total += loss.item() * len(n)
            means.append(total / len(loader.dataset))
        history.append({"epoch": epoch, "train_loss": means[0], "validation_loss": means[1]})
        if means[1] < best - 1e-5:
            best, best_weights, idle = means[1], deepcopy(net.state_dict()), 0
        else:
            idle += 1
            if idle >= cfg.patience:
                break
    
    net.load_state_dict(best_weights)
    net.eval()
    return net, history


@torch.inference_mode()
def features(prep, net, frame):
    n, c = prep.arrays(frame)
    embeddings, errors = [], []
    
    for start in range(0, len(n), net.cfg.batch_size):
        end = start + net.cfg.batch_size
        e, r = net(torch.from_numpy(n[start:end]), torch.from_numpy(c[start:end]))
        embeddings.append(e.numpy())
        errors.append(r.numpy())
    return prep.raw(n, c), np.concatenate(embeddings), np.concatenate(errors), (c == 0).sum(1)


def forest_input(raw, e, r, variant):
    if variant in ("RF_base", "RF_full"):
        return raw
    blocks = [raw]
    if variant in ("Hybrid", "RF_E"):
        blocks.append(sparse.csr_matrix(e))
    if variant in ("Hybrid", "RF_R"):
        blocks.append(sparse.csr_matrix(r[:, None]))
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def new_forest(cfg):
    return RandomForestClassifier(
        n_estimators=cfg.trees,
        max_depth=cfg.depth,
        min_samples_leaf=cfg.min_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=cfg.seed,
        n_jobs=cfg.threads,
    )


@dataclass
class Detector:
    prep: Preprocessor
    net: MixedAE
    forests: dict

    def predict(self, frame, variants):
        raw, e, r, oov = features(self.prep, self.net, frame)
        scores = {}
        for v in variants:
            if v == "AE_only":
                scores[v] = r
            else:
                scores[v] = self.forests[v].predict_proba(forest_input(raw, e, r, v))[:, 1]
        return scores, r, oov
