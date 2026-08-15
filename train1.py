import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import weight_norm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    fbeta_score, average_precision_score, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import optuna
import json
import shutil
import hashlib
from numpy.lib.format import open_memmap
from pytorch_lightning.callbacks import TQDMProgressBar
from copy import deepcopy
from pytorch_lightning.loggers import TensorBoardLogger
# torch.set_float32_matmul_precision('medium')

# 配置设备与实验标识
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "PatchDC"
# 由命令行 --seed 在主程序启动时赋值。保留为全局变量，供结果表和简报使用。
seed = None
EXPERIMENT_RUN_ID = None


def set_global_seed(seed_value):
    """统一设置 Python、NumPy、PyTorch 和 Lightning 的随机种子。"""
    seed_value = int(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # workers=True 使 DataLoader worker 也能基于该种子获得可复现的子种子。
    pl.seed_everything(seed_value, workers=True)


def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description=(
            "PatchDC 五折训练。使用 --seed 管理重复实验，"
            "并将每个 seed 的输出放入独立目录。"
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="本次完整五折实验的随机种子，默认 42。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "本次实验的完整输出目录。未指定时自动使用 "
            "./results/PatchDC/seed_<seed>。"
        ),
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 旧 CSV 的折内归一化配置
# -----------------------------------------------------------------------------
# 这些字段在旧 CSV 生成时已经分别除以 750 或 90.6。
# 训练/验证/测试读取时先乘回旧常数，再除以“当前折训练集”得到的最大值。
LEGACY_FEATURE_SCALES = {
    "1H_wmedian_range": 750.0,
    "1H_wmedian_iqr": 750.0,
    "data_mm_range": 750.0,
    "data_mm_iqr": 750.0,
    "1H_sum_max": 90.6,
}


def load_fold_normalization_params(json_path):
    """读取某一折仅由该折训练部分计算的归一化最大值。"""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"归一化 JSON 不存在：{json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    maxima = payload.get('maxima', payload)
    missing = [column for column in LEGACY_FEATURE_SCALES if column not in maxima]
    if missing:
        raise KeyError(f"{json_path} 缺少归一化字段：{missing}")

    params = {}
    for column in LEGACY_FEATURE_SCALES:
        value = float(maxima[column])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{json_path} 中 {column} 的训练最大值无效：{value}"
            )
        params[column] = value

    return params


# -----------------------------------------------------------------------------
# CSV 一次性缓存配置
# -----------------------------------------------------------------------------
# 仅用于消除每个 epoch / 每个 Optuna trial 反复执行 pd.read_csv 的开销。
# 缓存中保留旧 CSV 的原值；折内归一化仍在 __getitem__ 中使用当前折 JSON 实时完成。
CACHE_DATA100_COLUMNS = (
    "data_mm_jugy",
    "data_mm_iqr",
    "data_mm_range",
)
CACHE_DATA_COLUMNS = (
    "1H_sum_mima",
    "1H_wmedian_jugy",
    "1H_wmedian_mask",
    "1H_sum_max",
    "1H_wmedian_range",
    "1H_wmedian_iqr",
)
MVM_DATA_COLUMN = "1H_wmedian_mask"
MVM_DATA_CHANNEL_INDEX = CACHE_DATA_COLUMNS.index(MVM_DATA_COLUMN)
CACHE_DATA100_LENGTH = 100
CACHE_DATA_LENGTH = 24 * 15
CACHE_METADATA_FILENAME = "metadata.json"
CACHE_DATA100_FILENAME = "data100.npy"
CACHE_DATA_FILENAME = "data.npy"


def canonical_file_path(file_path):
    """生成跨训练、验证、测试清单一致的绝对规范路径。"""
    return os.path.normcase(
        os.path.abspath(os.path.normpath(file_path))
    )


def build_csv_cache_fingerprint(file_paths):
    """根据文件路径、大小和修改时间生成缓存指纹。"""
    digest = hashlib.sha256()
    for file_path in file_paths:
        stat_result = os.stat(file_path)
        digest.update(file_path.encode("utf-8", errors="surrogatepass"))
        digest.update(str(stat_result.st_size).encode("ascii"))
        digest.update(str(stat_result.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def prepare_csv_tensor_cache(mapping_groups, cache_dir):
    """
    将训练、验证和测试涉及的所有 CSV 各读取一次，写入两个连续的 .npy 文件。

    缓存内容：
      data100.npy: [N, 3, 100]
      data.npy:    [N, 6, 360]，时间方向与原 CSVDataset 完全一致（已 flip）

    如果源文件集合、文件大小和修改时间均未变化，则后续运行直接复用缓存。
    """
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # 去重但保持首次出现顺序。
    ordered_paths = []
    seen_paths = set()
    for mapping in mapping_groups:
        for file_path, _ in mapping:
            canonical_path = canonical_file_path(file_path)
            if canonical_path not in seen_paths:
                seen_paths.add(canonical_path)
                ordered_paths.append(canonical_path)

    if not ordered_paths:
        raise ValueError("没有可用于构建 CSV 缓存的文件")

    missing_files = [path for path in ordered_paths if not os.path.isfile(path)]
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"构建缓存时发现 {len(missing_files)} 个 CSV 不存在，前 10 个为：\n{preview}"
        )

    fingerprint = build_csv_cache_fingerprint(ordered_paths)
    metadata_path = os.path.join(cache_dir, CACHE_METADATA_FILENAME)
    data100_path = os.path.join(cache_dir, CACHE_DATA100_FILENAME)
    data_path = os.path.join(cache_dir, CACHE_DATA_FILENAME)

    # 已有缓存满足指纹和形状要求时直接复用。
    if (
        os.path.isfile(metadata_path)
        and os.path.isfile(data100_path)
        and os.path.isfile(data_path)
    ):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            cached_data100 = np.load(data100_path, mmap_mode="r")
            cached_data = np.load(data_path, mmap_mode="r")
            cache_is_valid = (
                metadata.get("fingerprint") == fingerprint
                and metadata.get("file_paths") == ordered_paths
                and tuple(cached_data100.shape)
                == (len(ordered_paths), len(CACHE_DATA100_COLUMNS), CACHE_DATA100_LENGTH)
                and tuple(cached_data.shape)
                == (len(ordered_paths), len(CACHE_DATA_COLUMNS), CACHE_DATA_LENGTH)
            )
            del cached_data100, cached_data
            if cache_is_valid:
                print(
                    f"✅ 复用 CSV 张量缓存：{cache_dir}，"
                    f"共 {len(ordered_paths)} 个样本"
                )
                return cache_dir
        except Exception as exc:
            print(f"⚠️ 现有缓存无效，将重新生成：{exc}")

    print(
        f"\n📦 首次构建 CSV 张量缓存：共 {len(ordered_paths)} 个文件。\n"
        "每个 CSV 只读取一次；后续 epoch、Optuna trial 和五折测试均不再读取 CSV。"
    )

    tmp_data100_path = data100_path + ".tmp.npy"
    tmp_data_path = data_path + ".tmp.npy"
    tmp_metadata_path = metadata_path + ".tmp"

    for tmp_path in (tmp_data100_path, tmp_data_path, tmp_metadata_path):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    data100_memmap = open_memmap(
        tmp_data100_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(ordered_paths), len(CACHE_DATA100_COLUMNS), CACHE_DATA100_LENGTH),
    )
    data_memmap = open_memmap(
        tmp_data_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(ordered_paths), len(CACHE_DATA_COLUMNS), CACHE_DATA_LENGTH),
    )

    required_columns = list(dict.fromkeys(
        list(CACHE_DATA100_COLUMNS) + list(CACHE_DATA_COLUMNS)
    ))
    cache_dtypes = {column: np.float32 for column in required_columns}

    try:
        for index, file_path in enumerate(ordered_paths):
            df = pd.read_csv(
                file_path,
                usecols=required_columns,
                dtype=cache_dtypes,
                engine="c",
            )

            if len(df) < CACHE_DATA_LENGTH:
                raise ValueError(
                    f"{file_path} 只有 {len(df)} 行，"
                    f"小时级分支至少需要 {CACHE_DATA_LENGTH} 行"
                )
            if len(df) < CACHE_DATA100_LENGTH:
                raise ValueError(
                    f"{file_path} 只有 {len(df)} 行，"
                    f"高频输入至少需要 {CACHE_DATA100_LENGTH} 行"
                )

            data100_values = np.stack(
                [
                    df[column].iloc[:CACHE_DATA100_LENGTH].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
                    for column in CACHE_DATA100_COLUMNS
                ],
                axis=0,
            )
            data_values = np.stack(
                [
                    df[column].iloc[:CACHE_DATA_LENGTH].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )[::-1]
                    for column in CACHE_DATA_COLUMNS
                ],
                axis=0,
            )

            data100_memmap[index] = data100_values
            data_memmap[index] = data_values

            if (index + 1) % 200 == 0 or index + 1 == len(ordered_paths):
                print(f"  已缓存 {index + 1}/{len(ordered_paths)} 个 CSV")

        data100_memmap.flush()
        data_memmap.flush()
        del data100_memmap, data_memmap

        metadata = {
            "cache_version": 1,
            "fingerprint": fingerprint,
            "file_count": len(ordered_paths),
            "file_paths": ordered_paths,
            "data100_columns": list(CACHE_DATA100_COLUMNS),
            "data_columns": list(CACHE_DATA_COLUMNS),
            "data100_shape": [
                len(ordered_paths),
                len(CACHE_DATA100_COLUMNS),
                CACHE_DATA100_LENGTH,
            ],
            "data_shape": [
                len(ordered_paths),
                len(CACHE_DATA_COLUMNS),
                CACHE_DATA_LENGTH,
            ],
            "dtype": "float32",
            "normalization_state": (
                "legacy CSV values are cached unchanged; fold-specific restoration "
                "and normalization are applied at read time"
            ),
        }
        with open(tmp_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        os.replace(tmp_data100_path, data100_path)
        os.replace(tmp_data_path, data_path)
        os.replace(tmp_metadata_path, metadata_path)

    except Exception:
        try:
            del data100_memmap, data_memmap
        except Exception:
            pass
        for tmp_path in (tmp_data100_path, tmp_data_path, tmp_metadata_path):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        raise

    print(f"✅ CSV 张量缓存构建完成：{cache_dir}")
    return cache_dir


# -----------------------------------------------------------------------------
# 独立测试集 Bootstrap 置信区间配置
# -----------------------------------------------------------------------------
TEST_THRESHOLD = 0.5
BOOTSTRAP_N_RESAMPLES = 10000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RANDOM_SEED = 20260715


def calculate_test_metrics(y_true, y_probability, threshold=TEST_THRESHOLD):
    """根据正类概率计算论文报告的五项测试指标。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_probability = np.asarray(y_probability, dtype=np.float64)

    if y_true.ndim != 1 or y_probability.ndim != 1:
        raise ValueError("y_true 和 y_probability 必须是一维数组")
    if len(y_true) != len(y_probability):
        raise ValueError("y_true 与 y_probability 长度不一致")
    if len(y_true) == 0:
        raise ValueError("测试样本为空")
    if not np.all(np.isfinite(y_probability)):
        raise ValueError("测试概率中存在 NaN 或无穷值")

    y_pred = (y_probability >= threshold).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_class_1": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f2_class_1": float(
            fbeta_score(
                y_true,
                y_pred,
                beta=2,
                pos_label=1,
                zero_division=0,
            )
        ),
        "aupr": float(average_precision_score(y_true, y_probability)),
    }



def calculate_confusion_counts(y_true, y_pred):
    """返回二分类混淆矩阵计数，正类标签为 1。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 与 y_pred 形状不一致")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def build_strategy_result_row(
    strategy_name,
    y_true,
    y_probability,
    threshold=TEST_THRESHOLD,
):
    """生成一种集成策略的指标、混淆矩阵和报警数量汇总。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_probability = np.asarray(y_probability, dtype=np.float64)
    y_pred = (y_probability >= threshold).astype(np.int64)
    metrics = calculate_test_metrics(y_true, y_probability, threshold=threshold)
    counts = calculate_confusion_counts(y_true, y_pred)

    return {
        "strategy": strategy_name,
        "threshold": float(threshold),
        "accuracy": metrics["accuracy"],
        "precision_class_1": metrics["precision_class_1"],
        "recall_class_1": metrics["recall_class_1"],
        "f2_class_1": metrics["f2_class_1"],
        "aupr": metrics["aupr"],
        "tp": counts["tp"],
        "tn": counts["tn"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "predicted_positive_count": int(np.sum(y_pred == 1)),
        "predicted_negative_count": int(np.sum(y_pred == 0)),
    }


def save_precision_recall_curve(
    y_true,
    y_probability,
    output_dir,
    file_prefix="ensemble_precision_recall_curve",
    highlighted_thresholds=(0.5, 0.4),
):
    """
    保存独立测试集的精确率-召回率曲线、曲线坐标和指定阈值工作点。

    主曲线基于五折模型的平均正类概率。阈值 0.5 和 0.4 的工作点
    被单独标记，以对应论文对决策阈值的讨论。
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_probability = np.asarray(y_probability, dtype=np.float64)
    os.makedirs(output_dir, exist_ok=True)

    precision_values, recall_values, thresholds = precision_recall_curve(
        y_true,
        y_probability,
        pos_label=1,
    )
    aupr = float(average_precision_score(y_true, y_probability))
    positive_prevalence = float(np.mean(y_true == 1))

    # sklearn 的 precision/recall 比 thresholds 多一个末端点。
    curve_thresholds = np.concatenate([thresholds, [np.nan]])
    curve_df = pd.DataFrame({
        "recall": recall_values,
        "precision": precision_values,
        "threshold": curve_thresholds,
    })
    curve_csv_path = os.path.join(output_dir, f"{file_prefix}_points.csv")
    curve_df.to_csv(curve_csv_path, index=False)

    operating_rows = []
    for threshold in highlighted_thresholds:
        threshold = float(threshold)
        y_pred = (y_probability >= threshold).astype(np.int64)
        counts = calculate_confusion_counts(y_true, y_pred)
        operating_rows.append({
            "threshold": threshold,
            "precision_class_1": float(
                precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            "recall_class_1": float(
                recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            "f2_class_1": float(
                fbeta_score(
                    y_true,
                    y_pred,
                    beta=2,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "tp": counts["tp"],
            "tn": counts["tn"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "predicted_positive_count": int(np.sum(y_pred == 1)),
        })

    operating_df = pd.DataFrame(operating_rows)
    operating_csv_path = os.path.join(
        output_dir,
        f"{file_prefix}_operating_points.csv",
    )
    operating_df.to_csv(operating_csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(
        recall_values,
        precision_values,
        linewidth=2.0,
        label=f"Five-fold mean-probability ensemble (AUPR={aupr:.4f})",
    )
    ax.axhline(
        positive_prevalence,
        linestyle="--",
        linewidth=1.2,
        label=f"Positive prevalence={positive_prevalence:.4f}",
    )

    for row in operating_rows:
        ax.scatter(
            row["recall_class_1"],
            row["precision_class_1"],
            s=55,
            label=(
                f"Threshold={row['threshold']:.1f}: "
                f"P={row['precision_class_1']:.3f}, "
                f"R={row['recall_class_1']:.3f}"
            ),
            zorder=3,
        )

    ax.set_xlabel("Recall of the true-warning class")
    ax.set_ylabel("Precision of the true-warning class")
    ax.set_title("Precision–recall curve on the independent test set")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    png_path = os.path.join(output_dir, f"{file_prefix}.png")
    pdf_path = os.path.join(output_dir, f"{file_prefix}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "png_path": png_path,
        "pdf_path": pdf_path,
        "curve_csv_path": curve_csv_path,
        "operating_csv_path": operating_csv_path,
        "aupr": aupr,
        "positive_prevalence": positive_prevalence,
    }


def stratified_bootstrap_confidence_intervals(
    y_true,
    y_probability,
    threshold=TEST_THRESHOLD,
    n_resamples=BOOTSTRAP_N_RESAMPLES,
    confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    random_seed=BOOTSTRAP_RANDOM_SEED,
):
    """
    对独立测试集执行按类别分层的非参数 Bootstrap。

    每次分别从正类和负类样本中有放回抽取与原类别数量相同的样本，
    因而每个重采样数据集仍包含原测试集相同数量的正类和负类。
    返回百分位法置信区间。
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_probability = np.asarray(y_probability, dtype=np.float64)

    if len(y_true) != len(y_probability):
        raise ValueError("y_true 与 y_probability 长度不一致")
    if n_resamples < 100:
        raise ValueError("n_resamples 至少应为 100；论文建议使用 10000")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须位于 0 和 1 之间")

    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        raise ValueError("Bootstrap 需要测试集中同时包含正类和负类")

    metric_names = [
        "accuracy",
        "precision_class_1",
        "recall_class_1",
        "f2_class_1",
        "aupr",
    ]
    samples = {name: np.empty(n_resamples, dtype=np.float64) for name in metric_names}
    rng = np.random.default_rng(random_seed)

    for bootstrap_idx in range(n_resamples):
        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )
        sampled_indices = np.concatenate([sampled_positive, sampled_negative])

        metrics = calculate_test_metrics(
            y_true[sampled_indices],
            y_probability[sampled_indices],
            threshold=threshold,
        )
        for name in metric_names:
            samples[name][bootstrap_idx] = metrics[name]

    point_estimates = calculate_test_metrics(
        y_true,
        y_probability,
        threshold=threshold,
    )

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0

    summary_rows = []
    for name in metric_names:
        lower = float(np.quantile(samples[name], lower_quantile))
        upper = float(np.quantile(samples[name], upper_quantile))
        estimate = point_estimates[name]
        summary_rows.append({
            "metric": name,
            "estimate": estimate,
            "ci_lower": lower,
            "ci_upper": upper,
            "estimate_percent": 100.0 * estimate,
            "ci_lower_percent": 100.0 * lower,
            "ci_upper_percent": 100.0 * upper,
            "confidence_level": confidence_level,
            "n_bootstrap": n_resamples,
            "bootstrap_method": "stratified nonparametric percentile bootstrap",
            "threshold": threshold,
            "random_seed": random_seed,
        })

    distribution_df = pd.DataFrame(samples)
    summary_df = pd.DataFrame(summary_rows)
    return point_estimates, summary_df, distribution_df


# -----------------------------------------------------------------------------
# 检查点空间管理与测试简报
# -----------------------------------------------------------------------------
def delete_checkpoint_files(root_dir, keep_path=None):
    """递归删除 root_dir 下的 .ckpt，仅保留 keep_path。"""
    if not root_dir or not os.path.exists(root_dir):
        return 0

    keep_abs = os.path.abspath(keep_path) if keep_path else None
    deleted = 0
    for current_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.ckpt'):
                continue
            path = os.path.abspath(os.path.join(current_root, filename))
            if keep_abs is not None and path == keep_abs:
                continue
            try:
                os.remove(path)
                deleted += 1
            except FileNotFoundError:
                pass
    return deleted


def update_fold_best_checkpoint(
    candidate_path,
    candidate_score,
    fold_save_dir,
    fold_idx,
    trial_number,
    checkpoint_state,
):
    """
    每完成一个 trial，只在当前 fold 根目录保留截至目前最优的一个检查点。

    当前 trial 自身的临时检查点会在函数末尾删除。最终每折固定保存：
        fold_X/best_model.ckpt
        fold_X/best_checkpoint_info.json
    """
    candidate_score = float(candidate_score)
    target_path = os.path.join(fold_save_dir, 'best_model.ckpt')
    current_best = float(checkpoint_state.get('score', -np.inf))

    if candidate_path and os.path.exists(candidate_path) and candidate_score > current_best:
        temporary_target = target_path + '.tmp'
        shutil.copy2(candidate_path, temporary_target)
        os.replace(temporary_target, target_path)

        checkpoint_state.clear()
        checkpoint_state.update({
            'path': target_path,
            'score': candidate_score,
            'trial_number': int(trial_number),
        })

        metadata = {
            'fold': int(fold_idx),
            'best_trial_number': int(trial_number),
            'selection_metric': 'positive_class_f2_on_validation_set',
            'best_validation_f2': candidate_score,
            'checkpoint_path': os.path.abspath(target_path),
            'retention_policy': 'only_one_best_checkpoint_per_fold',
        }
        with open(
            os.path.join(fold_save_dir, 'best_checkpoint_info.json'),
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(
            f"🏆 Fold {fold_idx} 更新全局最优检查点："
            f"trial={trial_number}, val-F2={candidate_score:.6f}"
        )

    # 无论当前 trial 是否成为最优，都删除 trial 目录中的检查点。
    # fold 根目录下的 best_model.ckpt 不在 candidate 的 trial 目录中。
    candidate_dir = os.path.dirname(candidate_path) if candidate_path else None
    deleted = delete_checkpoint_files(candidate_dir)
    if deleted:
        print(f"🧹 已清理当前 trial 的 {deleted} 个临时 ckpt")

    # 再次强制保证整个 fold 下只存在根目录的最优 ckpt。
    if checkpoint_state.get('path'):
        delete_checkpoint_files(fold_save_dir, keep_path=checkpoint_state['path'])

    return checkpoint_state.get('path')


def _metric_percent(value):
    return f"{100.0 * float(value):.2f}%"


def write_brief_test_report(
    output_dir,
    per_fold_results_df,
    fold_training_summary_df,
    strategy_comparison_df,
    bootstrap_summary_df,
    pr_curve_outputs,
    retained_checkpoint_paths,
    test_labels,
    threshold,
    output_files,
):
    """测试结束后生成简短 Markdown 和纯文本报告。"""
    os.makedirs(output_dir, exist_ok=True)

    test_labels = np.asarray(test_labels, dtype=np.int64)
    n_test = int(len(test_labels))
    n_positive = int(np.sum(test_labels == 1))
    n_negative = int(np.sum(test_labels == 0))

    lines = [
        '# PatchDC训练与独立测试简报',
        '',
        f'- 模型：{MODEL_NAME}',
        f'- 实验编号/随机种子：{EXPERIMENT_RUN_ID}',
        f'- 独立测试样本：{n_test}（正类 {n_positive}，负类 {n_negative}）',
        f'- 分类阈值：{float(threshold):.2f}',
        f'- Bootstrap：{BOOTSTRAP_N_RESAMPLES} 次，'
        f'{BOOTSTRAP_CONFIDENCE_LEVEL * 100:.0f}%置信区间',
        '',
        '## 1. 每折训练类别比例与权重',
        '',
        '| Fold | 训练负类 | 训练正类 | 负/正比例 | 正类损失权重 |',
        '|---:|---:|---:|---:|---:|',
    ]
    for _, row in fold_training_summary_df.sort_values('fold').iterrows():
        lines.append(
            f"| {int(row['fold'])} | {int(row['train_negative_count'])} | "
            f"{int(row['train_positive_count'])} | "
            f"{float(row['negative_to_positive_ratio']):.6f} | "
            f"{float(row['positive_class_weight']):.6f} |"
        )

    lines.extend([
        '',
        '## 2. 每折独立测试结果',
        '',
        '| Fold | Accuracy | Precision | Recall | F2 | AUPR |',
        '|---:|---:|---:|---:|---:|---:|',
    ])
    metric_columns = [
        'test_acc', 'test_precision_1', 'test_recall_1', 'test_f2', 'test_aupr'
    ]
    for _, row in per_fold_results_df.sort_values('fold').iterrows():
        lines.append(
            f"| {int(row['fold'])} | {_metric_percent(row['test_acc'])} | "
            f"{_metric_percent(row['test_precision_1'])} | "
            f"{_metric_percent(row['test_recall_1'])} | "
            f"{_metric_percent(row['test_f2'])} | "
            f"{_metric_percent(row['test_aupr'])} |"
        )

    if not per_fold_results_df.empty:
        means = per_fold_results_df[metric_columns].mean()
        stds = per_fold_results_df[metric_columns].std(ddof=1)
        lines.extend([
            '',
            '五折模型测试指标均值±标准差：',
            '',
            f"- Accuracy：{_metric_percent(means['test_acc'])} ± "
            f"{_metric_percent(stds['test_acc'])}",
            f"- Precision：{_metric_percent(means['test_precision_1'])} ± "
            f"{_metric_percent(stds['test_precision_1'])}",
            f"- Recall：{_metric_percent(means['test_recall_1'])} ± "
            f"{_metric_percent(stds['test_recall_1'])}",
            f"- F2：{_metric_percent(means['test_f2'])} ± "
            f"{_metric_percent(stds['test_f2'])}",
            f"- AUPR：{_metric_percent(means['test_aupr'])} ± "
            f"{_metric_percent(stds['test_aupr'])}",
        ])

    lines.extend([
        '',
        '## 3. 最终五折集成与一票赞成结果',
        '',
        '| 策略 | Accuracy | Precision | Recall | F2 | AUPR | TP | TN | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for _, row in strategy_comparison_df.iterrows():
        strategy_label = {
            'mean_probability_ensemble': '五折平均概率',
            'one_vote_positive': '一票赞成',
        }.get(row['strategy'], str(row['strategy']))
        lines.append(
            f"| {strategy_label} | {_metric_percent(row['accuracy'])} | "
            f"{_metric_percent(row['precision_class_1'])} | "
            f"{_metric_percent(row['recall_class_1'])} | "
            f"{_metric_percent(row['f2_class_1'])} | "
            f"{_metric_percent(row['aupr'])} | {int(row['tp'])} | "
            f"{int(row['tn'])} | {int(row['fp'])} | {int(row['fn'])} |"
        )

    lines.extend([
        '',
        '## 4. 五折平均概率集成的Bootstrap 95%置信区间',
        '',
        '| 指标 | 点估计 | 95% CI |',
        '|---|---:|---:|',
    ])
    display_names = {
        'accuracy': 'Accuracy',
        'precision_class_1': 'Precision',
        'recall_class_1': 'Recall',
        'f2_class_1': 'F2',
        'aupr': 'AUPR',
    }
    for _, row in bootstrap_summary_df.iterrows():
        lines.append(
            f"| {display_names.get(row['metric'], row['metric'])} | "
            f"{float(row['estimate_percent']):.2f}% | "
            f"{float(row['ci_lower_percent']):.2f}%–"
            f"{float(row['ci_upper_percent']):.2f}% |"
        )

    operating_df = pd.read_csv(pr_curve_outputs['operating_csv_path'])
    lines.extend([
        '',
        '## 5. Precision–Recall曲线',
        '',
        f"- AUPR：{_metric_percent(pr_curve_outputs['aupr'])}",
        f"- 测试集正类比例：{_metric_percent(pr_curve_outputs['positive_prevalence'])}",
    ])
    for _, row in operating_df.iterrows():
        lines.append(
            f"- 阈值 {float(row['threshold']):.1f}："
            f"Precision={_metric_percent(row['precision_class_1'])}，"
            f"Recall={_metric_percent(row['recall_class_1'])}，"
            f"F2={_metric_percent(row['f2_class_1'])}，"
            f"TP={int(row['tp'])}，FP={int(row['fp'])}，"
            f"TN={int(row['tn'])}，FN={int(row['fn'])}"
        )

    lines.extend([
        '',
        '## 6. 每折保留的唯一检查点',
        '',
    ])
    for fold_number, checkpoint_path in enumerate(retained_checkpoint_paths, start=1):
        lines.append(f'- Fold {fold_number}：`{os.path.abspath(checkpoint_path)}`')

    lines.extend([
        '',
        '## 7. 主要输出文件',
        '',
    ])
    for label, path in output_files.items():
        lines.append(f'- {label}：`{os.path.abspath(path)}`')

    lines.extend([
        '',
        '> 注：显著性检验需将本次生成的 significance_ready_metrics.csv '
        '与使用相同 seed 和 fold 的 PatchTST 结果配对后，运行 '
        'paired_ttest_wilcoxon.py。',
        '',
    ])

    markdown_text = '\n'.join(lines)
    markdown_path = os.path.join(output_dir, 'brief_test_report.md')
    text_path = os.path.join(output_dir, 'brief_test_report.txt')
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_text)
    with open(text_path, 'w', encoding='utf-8') as f:
        # 纯文本保留相同内容，便于直接打开和复制。
        f.write(markdown_text)

    return {'markdown_path': markdown_path, 'text_path': text_path}


# 1. 创建文件名-标签映射表
def get_sensor_id(filename):
    """按第一个下划线分割，左侧部分作为传感器编号。"""
    return os.path.basename(filename).split('_', 1)[0]


def get_binary_label(filename):
    """保持原标签规则：文件名末尾为 0 是负类，其余（1/2）是正类。"""
    label_str = os.path.splitext(os.path.basename(filename))[0].rsplit('_', 1)[-1]
    return 0 if label_str == '0' else 1


def create_file_label_mapping(txt_path, base_path):
    """读取普通 txt，创建（文件路径，标签）的元组列表，用于独立测试集。"""
    mapping = []
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, start=1):
            filename = line.strip()
            if not filename:
                continue
            try:
                mapping.append((os.path.join(base_path, filename), get_binary_label(filename)))
            except Exception as exc:
                raise ValueError(
                    f"{txt_path} 第 {line_no} 行文件名格式无效：{filename}"
                ) from exc
    return mapping


def create_grouped_fold_mapping(txt_path, base_path, n_splits=5):
    """
    读取格式为“fold_id\tfilename”的五折清单。

    fold_id 表示该文件在哪一折作为验证集；其余四折作为训练集。
    同一传感器只允许出现在一个 fold_id 中。
    """
    fold_mappings = {fold_id: [] for fold_id in range(1, n_splits + 1)}
    sensor_to_fold = {}

    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line_no == 1 and line.lower().startswith('fold_id'):
                continue

            fields = line.split('\t', 1)
            if len(fields) != 2:
                raise ValueError(
                    f"{txt_path} 第 {line_no} 行格式错误，应为：fold_id<制表符>filename；实际为：{line}"
                )

            fold_id = int(fields[0])
            filename = fields[1].strip()
            if fold_id not in fold_mappings:
                raise ValueError(
                    f"{txt_path} 第 {line_no} 行 fold_id={fold_id}，应位于 1~{n_splits}"
                )

            sensor_id = get_sensor_id(filename)
            previous_fold = sensor_to_fold.setdefault(sensor_id, fold_id)
            if previous_fold != fold_id:
                raise ValueError(
                    f"传感器 {sensor_id} 同时出现在 Fold {previous_fold} 和 Fold {fold_id}"
                )

            file_path = os.path.join(base_path, filename)
            fold_mappings[fold_id].append((file_path, get_binary_label(filename)))

    empty_folds = [fold_id for fold_id, items in fold_mappings.items() if not items]
    if empty_folds:
        raise ValueError(f"以下验证折没有样本：{empty_folds}")

    return fold_mappings


def verify_sensor_isolation(train_mapping, val_mapping, fold_idx):
    """运行前强制检查训练集和验证集没有共同传感器。"""
    train_sensors = {get_sensor_id(file_path) for file_path, _ in train_mapping}
    val_sensors = {get_sensor_id(file_path) for file_path, _ in val_mapping}
    overlap = sorted(train_sensors & val_sensors)
    if overlap:
        preview = ', '.join(overlap[:10])
        raise RuntimeError(
            f"Fold {fold_idx} 发生传感器泄漏，共 {len(overlap)} 个重叠传感器：{preview}"
        )
    return len(train_sensors), len(val_sensors)


# 2. 自定义数据集类
class CSVDataset(Dataset):
    """
    从合并后的 .npy 内存映射缓存读取样本，不再在每个 epoch 中读取 CSV。

    缓存保存的是旧 CSV 数值。五个统计特征仍在每次取样时按照当前折 JSON：
        旧值 × 750(或90.6) ÷ 当前折训练最大值
    因而只改变 I/O 方式，不改变输入数值和折内归一化逻辑。
    """

    def __init__(self, file_label_mapping, normalization_params, tensor_cache_dir):
        self.mapping = file_label_mapping
        self.normalization_params = normalization_params
        self.tensor_cache_dir = os.path.abspath(tensor_cache_dir)
        self.data100_path = os.path.join(
            self.tensor_cache_dir,
            CACHE_DATA100_FILENAME,
        )
        self.data_path = os.path.join(
            self.tensor_cache_dir,
            CACHE_DATA_FILENAME,
        )
        metadata_path = os.path.join(
            self.tensor_cache_dir,
            CACHE_METADATA_FILENAME,
        )

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"CSV 张量缓存元数据不存在：{metadata_path}。"
                "请先调用 prepare_csv_tensor_cache。"
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        path_to_cache_index = {
            canonical_file_path(file_path): index
            for index, file_path in enumerate(metadata["file_paths"])
        }

        self.cache_indices = []
        missing_paths = []
        for file_path, _ in self.mapping:
            canonical_path = canonical_file_path(file_path)
            cache_index = path_to_cache_index.get(canonical_path)
            if cache_index is None:
                missing_paths.append(canonical_path)
            else:
                self.cache_indices.append(cache_index)

        if missing_paths:
            preview = "\n".join(missing_paths[:10])
            raise KeyError(
                f"当前 mapping 中有 {len(missing_paths)} 个文件不在缓存中，"
                f"前 10 个为：\n{preview}"
            )

        # 不在主进程中持有 memmap；每个 DataLoader worker 首次取样时自行打开。
        # 这样在 Windows spawn 模式下不会把整个缓存复制到每个 worker。
        self._data100_memmap = None
        self._data_memmap = None

    def __len__(self):
        return len(self.mapping)

    def _ensure_cache_open(self):
        if self._data100_memmap is None:
            self._data100_memmap = np.load(
                self.data100_path,
                mmap_mode="r",
            )
        if self._data_memmap is None:
            self._data_memmap = np.load(
                self.data_path,
                mmap_mode="r",
            )

    def __getstate__(self):
        """DataLoader 多进程序列化时不传递已打开的 memmap 句柄。"""
        state = self.__dict__.copy()
        state["_data100_memmap"] = None
        state["_data_memmap"] = None
        return state

    def __getitem__(self, idx):
        file_path, label = self.mapping[idx]
        try:
            self._ensure_cache_open()
            cache_index = self.cache_indices[idx]

            # copy=True：后续折内归一化只修改当前样本副本，不改写磁盘缓存。
            slide_gy_array = np.array(
                self._data100_memmap[cache_index],
                dtype=np.float32,
                copy=True,
            )
            rain_gy_array = np.array(
                self._data_memmap[cache_index],
                dtype=np.float32,
                copy=True,
            )

            # 与原代码完全相同的折内归一化，只是直接作用于缓存数组通道。
            slide_gy_array[1] *= (
                LEGACY_FEATURE_SCALES["data_mm_iqr"]
                / self.normalization_params["data_mm_iqr"]
            )
            slide_gy_array[2] *= (
                LEGACY_FEATURE_SCALES["data_mm_range"]
                / self.normalization_params["data_mm_range"]
            )

            rain_gy_array[3] *= (
                LEGACY_FEATURE_SCALES["1H_sum_max"]
                / self.normalization_params["1H_sum_max"]
            )
            rain_gy_array[4] *= (
                LEGACY_FEATURE_SCALES["1H_wmedian_range"]
                / self.normalization_params["1H_wmedian_range"]
            )
            rain_gy_array[5] *= (
                LEGACY_FEATURE_SCALES["1H_wmedian_iqr"]
                / self.normalization_params["1H_wmedian_iqr"]
            )

            return {
                "data": torch.from_numpy(rain_gy_array),
                "data100": torch.from_numpy(slide_gy_array),
                "label": torch.tensor(label, dtype=torch.long),
            }
        except Exception as exc:
            print(f"Error loading cached sample {file_path}: {exc}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch], dim=0)

    labels = [item['label'] for item in batch]
    labels = torch.stack(labels, dim=0)

    return {
        'data': result['data'],
        'data100': result['data100'],
        'labels': labels
    }


import math
import numpy as np
from typing import Optional  # , Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def PositionalEncoding(q_len, hidden_size, normalize=True):
    pe = torch.zeros(q_len, hidden_size)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, hidden_size, exponential=False, normalize=True, eps=1e-3):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
                2
                * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
                * (torch.linspace(0, 1, hidden_size).reshape(1, -1) ** x)
                - 1
        )
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (
            2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1))
            - 1
    )
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, hidden_size):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty(
            (q_len, hidden_size)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "zeros":
        W_pos = torch.empty((q_len, hidden_size))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        W_pos = Coord2dPosEncoding(
            q_len, hidden_size, exponential=False, normalize=True
        )
    elif pe == "exp2d":
        W_pos = Coord2dPosEncoding(q_len, hidden_size, exponential=True, normalize=True)
    elif pe == "sincos":
        W_pos = PositionalEncoding(q_len, hidden_size, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class Flatten_Head(nn.Module):
    """
    Flatten_Head
    """

    def __init__(self, individual, n_vars, nf, h, c_out, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.c_out = c_out

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, h * c_out))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, h * c_out)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x hidden_size x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x hidden_size * patch_num]
                z = self.linears[i](z)  # z: [bs x h]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x h]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class _ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)
    """

    def __init__(
            self, hidden_size, n_heads, attn_dropout=0.0, res_attention=False, lsa=False, causal=True
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = hidden_size // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.causal = causal
    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        if self.causal:
            seq_len = q.size(2)
            # 创建下三角掩码（允许关注当前位置及之前位置）
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            if attn_mask is None:
                attn_mask = causal_mask
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """
        if not self.res_attention:
            # Use torch's built-in flash attention for efficient computation
            # Note: This will not return attention weights/scores
            # The shapes of q, k, v must be: [batch, n_heads, seq_len, head_dim]
            # Reshape q, k, v into [batch*n_heads, seq_len, head_dim] as required by torch.nn.functional.scaled_dot_product_attention
            bs, n_heads, seq_len, head_dim = q.shape
            q_ = q.reshape(bs * n_heads, seq_len, head_dim)
            k_ = k.permute(0, 1, 3, 2).reshape(bs * n_heads, seq_len, head_dim)
            v_ = v.reshape(bs * n_heads, seq_len, head_dim)
            # If attn_mask exists, convert it to the appropriate format for flash attention (e.g. [batch*n_heads, seq_len, seq_len])
            if attn_mask is not None:
                attn_mask = attn_mask.repeat(bs * n_heads, 1, 1)
            output = F.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p,
                is_causal=False,
            )
            # Restore the original shape
            output = output.reshape(bs, n_heads, seq_len, head_dim)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return output, None
        else:
            # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
            attn_scores = (
                    torch.matmul(q, k) * self.scale
            )  # attn_scores : [bs x n_heads x max_q_len x q_len]

            # Add pre-softmax attention scores from the previous layer (optional)
            if prev is not None:
                attn_scores = attn_scores + prev

            # Attention mask (optional)
            if (
                    attn_mask is not None
            ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
                if attn_mask.dtype == torch.bool:
                    attn_scores.masked_fill_(attn_mask, -np.inf)
                else:
                    attn_scores += attn_mask

            # Key padding mask (optional)
            if (
                    key_padding_mask is not None
            ):  # mask with shape [bs x q_len] (only when max_w_len == q_len)
                attn_scores.masked_fill_(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
                )

            # normalize the attention weights
            attn_weights = F.softmax(
                attn_scores, dim=-1
            )  # attn_weights   : [bs x n_heads x max_q_len x q_len]
            attn_weights = self.attn_dropout(attn_weights)

            # compute the new values given the attention weights
            output = torch.matmul(
                attn_weights, v
            )  # output: [bs x n_heads x max_q_len x d_v]

            return output, attn_weights, attn_scores


class _MultiheadAttention(nn.Module):
    """
    _MultiheadAttention
    """

    def __init__(
            self,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            res_attention=False,
            attn_dropout=0.0,
            proj_dropout=0.0,
            qkv_bias=True,
            lsa=False,
    ):
        """
        Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x hidden_size]
            K, V:    [batch_size (bs) x q_len x hidden_size]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = hidden_size // n_heads if d_k is None else d_k
        d_v = hidden_size // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(hidden_size, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(hidden_size, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(hidden_size, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            hidden_size,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, hidden_size), nn.Dropout(proj_dropout)
        )

    def forward(
            self,
            Q: torch.Tensor,
            K: Optional[torch.Tensor] = None,
            V: Optional[torch.Tensor] = None,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = (
            self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = (
            self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        )  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = (
            self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        )  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class Transpose(nn.Module):
    """
    Transpose
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class TSTEncoder(nn.Module):
    """
    TSTEncoder
    """

    def __init__(
            self,
            q_len,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            linear_hidden_size=None,
            norm="BatchNorm",
            attn_dropout=0.0,
            dropout=0.0,
            activation="gelu",
            res_attention=False,
            n_layers=1,
            pre_norm=False,
            store_attn=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    hidden_size,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    linear_hidden_size=linear_hidden_size,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
            self,
            src: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        output = src
        scores = None
        if self.res_attention: #false
            for mod in self.layers:
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            return output


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


class TSTEncoderLayer(nn.Module):
    """
    TSTEncoderLayer
    """

    def __init__(
            self,
            q_len,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            linear_hidden_size=256,
            store_attn=False,
            norm="BatchNorm",
            attn_dropout=0,
            dropout=0.0,
            bias=True,
            activation="gelu",
            res_attention=False,
            pre_norm=False,
    ):
        super().__init__()
        assert (
            not hidden_size % n_heads
        ), f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
        d_k = hidden_size // n_heads if d_k is None else d_k
        d_v = hidden_size // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            hidden_size,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(hidden_size)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, linear_hidden_size, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(linear_hidden_size, hidden_size, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(hidden_size)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
            self,
            src: torch.Tensor,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):  # -> Tuple[torch.Tensor, Any]:

        # Multi-Head attention sublayer
        if self.pre_norm:#False
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention: #False
            src2, attn, scores = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:#False
            return src, scores
        else:
            return src


class TSTiEncoder(nn.Module):  # i means channel-independent
    """
    TSTiEncoder
    """

    def __init__(
            self,
            c_in,
            patch_num,
            patch_len,
            max_seq_len=1024,
            n_layers=3,
            hidden_size=128,
            n_heads=16,
            d_k=None,
            d_v=None,
            linear_hidden_size=256,
            norm="BatchNorm",
            attn_dropout=0.0,
            dropout=0.0,
            act="gelu",
            store_attn=False,
            key_padding_mask="auto",
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            pe="zeros",
            learn_pe=True,
    ):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, hidden_size
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, hidden_size)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            q_len,
            hidden_size,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def forward(self, x) -> torch.Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x hidden_size]

        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # u: [bs * nvars x patch_num x hidden_size]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x hidden_size]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x hidden_size]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x hidden_size]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x hidden_size x patch_num]

        return z


class RevIN(nn.Module):
    """RevIN (Reversible-Instance-Normalization)"""

    def __init__(
            self,
            num_features: int,
            eps=1e-5,
            affine=False,
            subtract_last=False,
            non_norm=False,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param substract_last: if True, the substraction is based on the last value
                               instead of the mean in normalization
        :param non_norm: if True, no normalization performed.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class PatchTST_backbone(nn.Module):
    """
    PatchTST_backbone
    """

    def __init__(
            self,
            c_in: int,
            c_out: int,
            input_size: int,
            h: int,
            patch_len: int,
            stride: int,
            max_seq_len: Optional[int] = 1024,
            n_layers: int = 3,
            hidden_size=128,
            n_heads=16,
            d_k: Optional[int] = None,
            d_v: Optional[int] = None,
            linear_hidden_size: int = 256,
            norm: str = "BatchNorm",
            attn_dropout: float = 0.0,
            dropout: float = 0.0,
            act: str = "gelu",
            key_padding_mask: str = "auto",
            padding_var: Optional[int] = None,
            attn_mask: Optional[torch.Tensor] = None,
            res_attention: bool = True,
            pre_norm: bool = False,
            store_attn: bool = False,
            pe: str = "zeros",
            learn_pe: bool = True,
            fc_dropout: float = 0.0,
            head_dropout=0,
            padding_patch=None,
            pretrain_head: bool = False,
            head_type="flatten",
            individual=False,
            revin=True,
            affine=True,
            subtract_last=False,
    ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((input_size - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case #None
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
        )

        # Head
        self.head_nf = hidden_size * patch_num
        self.n_vars = c_in
        self.c_out = c_out
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs
        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                self.n_vars,
                self.head_nf,
                h,
                c_out,
                head_dropout=head_dropout,
            )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x hidden_size x patch_num]
        z = self.head(z)  # z: [bs x nvars x h]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class ConvNet(nn.Module):
    def __init__(self, c_in1=3, c_in2=6, input_size1=100, input_size2=360,
                 hidden_size=128, num_classes=2, encoder_layers=3, n_heads=8,
                 patch_len=16, stride=8, revin=False, dropout=0.1):
        super().__init__()
        # MVM 融合特征保持原样：前 3 个通道在每个时间步融合为 1 个通道。
        self.fc1 = nn.Linear(3, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

        # data100 分支完整保留，不对其 3 个通道做消融。
        self.branch1 = PatchTST_backbone(
            c_in=c_in1,
            c_out=1,
            input_size=input_size1,
            h=hidden_size,
            patch_len=patch_len,
            stride=stride,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            revin=revin,
            head_type='flatten',
            individual=False,
            dropout=dropout
        )

        # MVM 消融后的 branch2：5 个非 MVM 原始通道 + 1 个 MVM fused feature。
        self.branch2 = PatchTST_backbone(
            c_in=c_in2,
            c_out=1,
            input_size=input_size2,
            h=hidden_size,
            patch_len=patch_len,
            stride=stride,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            revin=revin,
            head_type='flatten',
            individual=False,
            dropout=dropout
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(c_in1 + c_in2, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # data100 分支保持原样：3 个通道全部参与。
        out1 = self.branch1(x['data100'])

        # fused feature 先按原逻辑生成，其输入仍包含 1H_wmedian_mask（MVM）。
        fused_feature = self.fc2(
            self.relu(self.fc1(x['data'][:, :3, :].permute(0, 2, 1)))
        ).permute(0, 2, 1)

        # 只在 branch2 的独立原始特征中剔除 1H_wmedian_mask（索引 2）。
        data_without_mvm = torch.cat(
            [
                x['data'][:, :MVM_DATA_CHANNEL_INDEX, :],
                x['data'][:, MVM_DATA_CHANNEL_INDEX + 1:, :],
            ],
            dim=1,
        )
        branch2_input = torch.cat([data_without_mvm, fused_feature], dim=1)
        out2 = self.branch2(branch2_input)
        features = torch.cat([out1, out2], dim=1)
        features = self.dropout(features)
        features = torch.mean(features, dim=2)
        logits = self.classifier(features)
        return logits

class CustomProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        # 返回一个禁用的进度条
        bar = super().init_validation_tqdm()
        bar.disable = True  # 禁用验证进度条
        return bar


# PyTorch Lightning 数据模块 (修复了持久化工作进程问题)
class RainfallDataModule(pl.LightningDataModule):
    def __init__(self, train_mapping, val_mapping, test_mapping, normalization_params, tensor_cache_dir, batch_size=200, num_workers=4):
        super().__init__()
        self.train_mapping = train_mapping
        self.val_mapping = val_mapping
        self.test_mapping = test_mapping
        self.normalization_params = normalization_params
        self.tensor_cache_dir = tensor_cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 三个数据集使用同一套“当前折训练集”归一化参数。
        self.train_dataset = CSVDataset(
            self.train_mapping, self.normalization_params, self.tensor_cache_dir
        )
        self.val_dataset = CSVDataset(
            self.val_mapping, self.normalization_params, self.tensor_cache_dir
        )
        self.test_dataset = CSVDataset(
            self.test_mapping, self.normalization_params, self.tensor_cache_dir
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0

        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )


class LossStabilityCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics["train_loss_epoch"]
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        self.loss_history.append(current_loss.item())


# PyTorch Lightning 模型模块
class RainfallModel(pl.LightningModule):
    def __init__(self, lr=0.001, dropout=0.2, hidden_size=None, encoder_layers=None, n_heads=None, patch_len=None, stride=None,
                 class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConvNet(
            dropout=dropout,
            hidden_size=hidden_size,
            encoder_layers=encoder_layers,
            n_heads=n_heads,
            patch_len=patch_len,
            stride=stride,
        )

        self.class_weights = class_weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.lr = lr

        # 用于存储每折的验证结果
        self.validation_step_outputs = []
        # 用于测试
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # 计算训练准确率
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # 计算验证准确率
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        # 存储结果用于epoch结束时的计算
        results = {
            'val_loss': loss,
            'val_acc': acc,
            'labels': labels,
            'preds': preds
        }
        self.validation_step_outputs.append(results)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return results

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return

        # 使用与重新评估相同的计算方式
        all_labels = torch.cat([x['labels'] for x in outputs])
        all_preds = torch.cat([x['preds'] for x in outputs])
        print("all_preds:", len(all_preds))
        # 添加zero_division参数确保一致性
        precision_1 = precision_score(all_labels.cpu().numpy(),
                                      all_preds.cpu().numpy(),
                                      pos_label=1,
                                      zero_division=0)

        recall_1 = recall_score(all_labels.cpu().numpy(),
                                all_preds.cpu().numpy(),
                                pos_label=1,
                                zero_division=0)

        f1_1 = fbeta_score(all_labels.cpu().numpy(),
                 all_preds.cpu().numpy(),
                 beta=2,  # 添加这个关键参数
                 pos_label=1,
                 zero_division=0)

        # 存储到模块属性中
        self.avg_val_precision_1 = precision_1
        self.avg_val_recall_1 = recall_1
        self.avg_val_f1 = f1_1
    

        self.log('avg_val_precision_1', precision_1, prog_bar=True)
        self.log('avg_val_recall_1', recall_1, prog_bar=True)
        self.log('avg_val_f1', f1_1, prog_bar=True)
        
        print(" call_current_f1", f1_1)

        self.validation_step_outputs.clear()
        return {'val_precision_1': precision_1, 'val_recall_1': recall_1, 'val_f1': f1_1}

    def test_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']

        outputs = self(inputs)
        probs = torch.softmax(outputs, dim=1)
        pos_probs = probs[:, 1]  # 正类概率
        preds = (pos_probs >= 0.5).long()  # 根据阈值0.5进行二值化

        result = {'labels': labels, 'preds': preds}
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        # 聚合所有测试步骤的结果
        labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        preds = torch.cat([x['preds'] for x in self.test_step_outputs])

        precision_1 = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), pos_label=1, zero_division=0)
        # 计算召回率
        recall_1 = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), pos_label=1, zero_division=0)
        # 记录召回率
        self.log('test_precision_1', precision_1, 'test_recall_1', recall_1, prog_bar=True)

        # 清空
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-3
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 自定义回调函数，用于在满足召回率条件下选择F1最高的模型
class RecallConditionalModelCheckpoint(pl.Callback):
    def __init__(self, recall_threshold=0.85, val_mapping=None, save_dir=None):
        super().__init__()
        self.recall_threshold = recall_threshold
        self.save_dir = save_dir or os.getcwd()
        os.makedirs(self.save_dir, exist_ok=True)

        # 存储当前epoch状态
        self.current_epoch_state = None
        self.best_model_path = None

        # 最佳模型指标
        self.best_f1 = 0.0
        self.best_recall = 0.0
        self.best_epoch = -1
        self.val_mapping = val_mapping

    def on_train_epoch_end(self, trainer, pl_module):
        """保存当前训练结束时的模型状态"""
        # 深拷贝模型状态
        model_state = deepcopy(pl_module.state_dict())

        # 获取当前优化器状态
        optimizer_state = None
        if trainer.optimizers:
            optimizer_state = deepcopy(trainer.optimizers[0].state_dict())

        # 获取Lightning版本
        from pytorch_lightning import __version__ as lightning_version

        # 构建完整状态字典
        self.current_epoch_state = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'pytorch-lightning_version': lightning_version,
            'state_dict': model_state,
            'optimizer_states': [optimizer_state] if optimizer_state else [],
            'lr_schedulers': [],
            'callbacks': {},  # 空回调状态
            'hparams_name': 'hparams',  # 必需字段
            'hyper_parameters': pl_module.hparams
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        """验证结束时更新最佳模型"""
        # 1. 获取当前验证指标
        metrics = trainer.callback_metrics
        current_recall = metrics.get('avg_val_recall_1', 0.0)
        current_f1 = metrics.get('avg_val_f1', 0.0)
        epoch = trainer.current_epoch

        # 2. 更新最佳模型条件
        should_save = False
        if current_recall >= self.recall_threshold:
            if current_f1 > self.best_f1:
                should_save = True
                self.best_f1 = current_f1
                self.best_recall = current_recall
                self.best_epoch = epoch

        # 3. 保存最佳模型
        if should_save and self.current_epoch_state:
            # 准备保存路径
            filename = f"best_epoch={epoch}_recall={current_recall:.4f}_f1={current_f1:.4f}.ckpt"
            best_candidate_path = os.path.join(self.save_dir, filename)

            # 保存当前 trial 的最新最优检查点，并立即删除该 trial 的旧检查点。
            previous_best_path = self.best_model_path
            torch.save(self.current_epoch_state, best_candidate_path)
            self.best_model_path = best_candidate_path
            if (
                previous_best_path
                and previous_best_path != best_candidate_path
                and os.path.exists(previous_best_path)
            ):
                os.remove(previous_best_path)
            print(f"💾 保存当前 trial 最佳模型到: {best_candidate_path}")

            # # 验证模型加载功能
            # try:
            #     # 注意这里不能直接使用RainfallModel.load_from_checkpoint
            #     # 因为加载需要初始化模型参数
            #     checkpoint = torch.load(best_candidate_path)
            #     state_dict = checkpoint['state_dict']
            #     # 创建模型实例并加载状态
            #     model = deepcopy(pl_module)
            #     model.load_state_dict(state_dict)
            #     model.to(device)
            #     model.eval()
            #     print("cc:", self.best_model_path)
            #
            #     # 创建测试数据集
            #     val_dataset = CSVDataset(self.val_mapping)
            #     print("val_dataset", len(val_dataset))
            #     val_loader = DataLoader(
            #         val_dataset,
            #         batch_size=40,
            #         shuffle=False,
            #         num_workers=4,
            #         collate_fn=collate_fn,
            #         persistent_workers=True,
            #         pin_memory=True
            #     )
            #
            #     # 评估模型在验证集上的表现
            #     all_labels = []
            #     all_preds = []
            #
            #     with torch.no_grad():
            #         for batch in val_loader:
            #             if batch is None:
            #                 continue
            #
            #             inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            #             labels = batch['labels'].to(device)
            #
            #             outputs = model(inputs)
            #             _, preds = torch.max(outputs, 1)
            #
            #             all_labels.extend(labels.cpu().numpy())
            #             all_preds.extend(preds.cpu().numpy())
            #
            #     # 计算验证集上的评估指标
            #     precision_1 = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
            #     recall_1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            #     f11 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
            #     f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            #     ac = accuracy_score(all_labels, all_preds)
            #     print(f"验证集cc：f1: {f1},精确度：{precision_1:.4f}，召回率: {recall_1:.4f}，F1_1分数: {f11:.4f}")
            #     print("✅ 模型状态加载验证成功")
            # except Exception as e:
            #     print(f"⚠️ 模型状态加载失败: {e}")
            #
            #
            #

    def on_train_end(self, trainer, pl_module):
        """训练结束时总结最佳模型"""
        if self.best_model_path:
            print("\n🏆 训练完成 - 最佳模型总结:")
            print(f"   - 路径: {self.best_model_path}")
            print(f"   - Epoch: {self.best_epoch}")
            print(f"   - 召回率: {self.best_recall:.4f}")
            print(f"   - F1分数: {self.best_f1:.4f}")
        else:
            print("⚠️ 未找到满足条件的模型")

# 贝叶斯优化目标函数
def objective(
    trial,
    train_mapping,
    val_mapping,
    test_mapping,
    normalization_params,
    tensor_cache_dir,
    fold_idx,
    save_dir,
    fold_checkpoint_state,
):
    # 定义超参数搜索空间
    hidden_size = trial.suggest_categorical("hidden_size", [32,64, 128]) #C
    encoder_layers = trial.suggest_int("encoder_layers",  1, 3)#C
    n_heads = trial.suggest_categorical("n_heads", [2, 4,8])#C
    patch_len = trial.suggest_int("patch_len", 8, 24, step=4)
    stride = trial.suggest_int("stride", 3, 7, step=2)#C

    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    dropout = trial.suggest_categorical('dropout', [0.1, 0.3, 0.5])
    batch_size = trial.suggest_categorical('batch_size', [100])
    # Comment 11：正类损失权重直接由当前折训练部分的类别比例确定。
    # 负类权重保持为 1，正类权重为 N_negative / N_positive。
    train_negative_count = sum(label == 0 for _, label in train_mapping)
    train_positive_count = sum(label == 1 for _, label in train_mapping)
    if train_positive_count <= 0:
        raise ValueError(f"Fold {fold_idx} 训练集中没有正类样本，无法计算类别权重")
    positive_class_weight = train_negative_count / train_positive_count
    class_weights = [1.0, float(positive_class_weight)]
    trial.set_user_attr('train_negative_count', int(train_negative_count))
    trial.set_user_attr('train_positive_count', int(train_positive_count))
    trial.set_user_attr('positive_class_weight', float(positive_class_weight))
    # 创建数据模块
    data_module = RainfallDataModule(
        train_mapping=train_mapping,
        val_mapping=val_mapping,
        test_mapping=test_mapping,
        normalization_params=normalization_params,
        tensor_cache_dir=tensor_cache_dir,
        batch_size=batch_size
    )

    # 创建模型
    model = RainfallModel(
        lr=lr,
        dropout=dropout,
        hidden_size=hidden_size,
        encoder_layers=encoder_layers,
        n_heads=n_heads,
        patch_len=patch_len,
        stride=stride,
        class_weights=class_weights
    )

    # 创建日志和模型保存路径
    # logger = CSVLogger(save_dir, name=f"fold_{fold_idx}")
    logger = TensorBoardLogger(
        save_dir=os.path.join(save_dir, f"fold_{fold_idx}"),
        name=f"trial_{trial.number}",
        default_hp_metric=False  # 避免重复记录超参数
    )
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 自定义回调函数 - 在满足召回率条件下选择F1最高的模型
    recall_checkpoint = RecallConditionalModelCheckpoint(
        recall_threshold=0.8,
        val_mapping=val_mapping,
    save_dir = checkpoint_dir
    )

    # 早停回调 - 监控召回率
    early_stop_callback = EarlyStopping(
        monitor='avg_val_f1',
        min_delta=0.001,
        patience=40,
        verbose=True,
        mode='max'
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_model-{epoch}-{avg_val_f1:.4f}',
        monitor='avg_val_f1',
        mode='max',
        save_top_k=1
    )
    progress_bar = CustomProgressBar()
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=200,
        logger=logger,
        callbacks=[LossStabilityCallback(), recall_checkpoint, early_stop_callback, progress_bar],
        enable_progress_bar=True,
        log_every_n_steps=5,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[0],

    )

    # 训练模型
    trainer.fit(model, datamodule=data_module)

    # 获取最佳模型路径
    try:
        best_model_path = recall_checkpoint.best_model_path or recall_checkpoint.best_f1_model_path
    except:
        best_model_path = None

    # 手动加载最佳模型并在测试集上评估
    if best_model_path:
        print(f"📦 加载最佳模型: {best_model_path}")
        model = RainfallModel.load_from_checkpoint(best_model_path)
        model.to(device)
        model.eval()

        # 创建测试数据集
        val_dataset = CSVDataset(
            val_mapping, normalization_params, tensor_cache_dir
        )
        print("val_dataset", len(val_dataset))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True
        )

        # 评估模型在验证集上的表现
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # 计算验证集上的评估指标
        precision_1 = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        recall_1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f2_1 = fbeta_score(
            all_labels,
            all_preds,
            beta=2,
            pos_label=1,
            zero_division=0,
        )
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        ac = accuracy_score(all_labels, all_preds)
        print(
            f"验证集：macro-F1={macro_f1:.4f}，准确率={ac:.4f}，"
            f"正类精确率={precision_1:.4f}，正类召回率={recall_1:.4f}，"
            f"正类F2={f2_1:.4f}"
        )

        # 空间控制：当前 trial 结束后，仅保留本折截至目前最优的一个 ckpt。
        update_fold_best_checkpoint(
            candidate_path=best_model_path,
            candidate_score=f2_1,
            fold_save_dir=save_dir,
            fold_idx=fold_idx,
            trial_number=trial.number,
            checkpoint_state=fold_checkpoint_state,
        )
        return f2_1

    else:
        print("❌ 未找到最佳模型路径")
        # 防止异常 trial 留下残余检查点。
        delete_checkpoint_files(checkpoint_dir)
        return 0.0


# 主程序
if __name__ == "__main__":
    args = parse_command_line_args()
    seed = int(args.seed)
    EXPERIMENT_RUN_ID = seed
    set_global_seed(seed)

    # 配置路径
    base_path ="../trainall"
    fold_manifest_path = "train1_grouped_5fold.txt"
    test_txt_path = "test.txt"
    # 先运行 make_fold_normalization_json_from_legacy_csv.py 生成此目录。
    normalization_json_dir = "fold_normalization_json"
    # 第一次运行时生成约几十 MB 的 .npy 缓存；后续运行直接复用。
    csv_tensor_cache_dir = os.path.join(base_path, "_patchdc_tensor_cache")
    n_splits = 5

    # 从预先生成的传感器分组五折清单读取数据。
    # fold_id 对应验证折，其余四折自动合并为训练集。
    fold_mappings = create_grouped_fold_mapping(
        fold_manifest_path, base_path, n_splits=n_splits
    )
    test_mapping = create_file_label_mapping(test_txt_path, base_path)

    # 训练开始前一次性读取全部 CSV。之后所有折、trial、epoch 和测试只读取 .npy memmap。
    cache_mapping_groups = [
        fold_mappings[fold_id]
        for fold_id in range(1, n_splits + 1)
    ] + [test_mapping]
    prepare_csv_tensor_cache(
        cache_mapping_groups,
        csv_tensor_cache_dir,
    )

    fold_results = []
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.output_dir:
        base_save_dir = os.path.abspath(args.output_dir)
    else:
        base_save_dir = os.path.abspath(
            os.path.join('./results', MODEL_NAME, f'seed_{seed}')
        )

    # 每个 seed 使用独立目录，避免不同重复实验的模型和结果互相混合。
    # 已完成的目录由运行脚本自动跳过；若目录中已有残留文件则拒绝覆盖。
    if os.path.isdir(base_save_dir) and os.listdir(base_save_dir):
        raise FileExistsError(
            f"输出目录已存在且非空：{base_save_dir}。"
            "请确认该 seed 是否已完成；若是中断残留，请先移动或删除该目录。"
        )
    os.makedirs(base_save_dir, exist_ok=True)

    run_metadata_path = os.path.join(base_save_dir, 'run_metadata.json')
    run_metadata = {
        'model_name': MODEL_NAME,
        'seed': seed,
        'run_id': EXPERIMENT_RUN_ID,
        'start_time': current_time,
        'base_path': os.path.abspath(base_path),
        'fold_manifest_path': os.path.abspath(fold_manifest_path),
        'test_txt_path': os.path.abspath(test_txt_path),
        'normalization_json_dir': os.path.abspath(normalization_json_dir),
        'output_dir': base_save_dir,
        # 物理 GPU 编号由运行脚本通过 PATCHDC_PHYSICAL_GPU_ID 传入。
        # 使用 CUDA_VISIBLE_DEVICES 后，PyTorch 内部通常会把所选物理 GPU 映射为 cuda:0。
        'physical_gpu_id': os.environ.get('PATCHDC_PHYSICAL_GPU_ID'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_cuda_available': bool(torch.cuda.is_available()),
        'torch_visible_gpu_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        'torch_current_device': int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        'torch_current_device_name': (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
    }
    with open(run_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    print(f"本次模型: {MODEL_NAME}")
    print(f"本次随机种子: {seed}")
    print(f"结果目录: {base_save_dir}")
    print(
        "物理 GPU 编号: "
        f"{os.environ.get('PATCHDC_PHYSICAL_GPU_ID', '未指定')}；"
        "CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '未限制')}"
    )
    if torch.cuda.is_available():
        print(
            f"PyTorch 可见 GPU 数量: {torch.cuda.device_count()}；"
            f"当前逻辑设备: cuda:{torch.cuda.current_device()}；"
            f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    else:
        print("未检测到可用 CUDA，训练将使用 CPU。")

    # 存储所有折的最佳模型路径及训练类别比例，用于最终简报。
    best_models = []
    fold_training_summaries = []
    for fold_idx in range(1, n_splits + 1):
        print(f"\n🚀 开始第 {fold_idx}/{n_splits} 折训练")

        # 当前 fold_id 的样本作为验证集，其余 fold_id 的样本作为训练集。
        fold_val_mapping = list(fold_mappings[fold_idx])
        fold_train_mapping = [
            item
            for current_fold, items in fold_mappings.items()
            if current_fold != fold_idx
            for item in items
        ]

        # 硬性检查：同一传感器不能同时出现在训练集和验证集中。
        train_sensor_count, val_sensor_count = verify_sensor_isolation(
            fold_train_mapping, fold_val_mapping, fold_idx
        )
        train_pos = sum(label == 1 for _, label in fold_train_mapping)
        train_neg = sum(label == 0 for _, label in fold_train_mapping)
        val_pos = sum(label == 1 for _, label in fold_val_mapping)
        val_neg = sum(label == 0 for _, label in fold_val_mapping)
        if train_pos <= 0:
            raise ValueError(f"Fold {fold_idx} 训练集中没有正类样本")
        positive_class_weight = train_neg / train_pos
        print(
            f"Fold {fold_idx}: 训练样本={len(fold_train_mapping)} "
            f"(负类={train_neg}, 正类={train_pos}, "
            f"负/正={positive_class_weight:.6f}, 传感器={train_sensor_count})；"
            f"验证样本={len(fold_val_mapping)} "
            f"(负类={val_neg}, 正类={val_pos}, 传感器={val_sensor_count})"
        )
        print(
            f"Fold {fold_idx} 交叉熵类别权重："
            f"negative=1.0, positive={positive_class_weight:.6f}"
        )
        fold_training_summaries.append({
            'fold': int(fold_idx),
            'train_negative_count': int(train_neg),
            'train_positive_count': int(train_pos),
            'validation_negative_count': int(val_neg),
            'validation_positive_count': int(val_pos),
            'negative_to_positive_ratio': float(positive_class_weight),
            'positive_class_weight': float(positive_class_weight),
        })

        # 读取当前折专属参数：该 JSON 只能由其余四折训练数据计算。
        normalization_json_path = os.path.join(
            normalization_json_dir,
            f"fold_{fold_idx}_normalization.json"
        )
        normalization_params = load_fold_normalization_params(
            normalization_json_path
        )
        print(f"Fold {fold_idx} 使用归一化参数：{normalization_json_path}")
        for column, value in normalization_params.items():
            print(f"  {column}: {value:.10g}")

        # 创建当前折的保存目录
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_idx}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # 同时复制该折归一化 JSON 到结果目录，确保模型与参数成对保存。
        shutil.copy2(
            normalization_json_path,
            os.path.join(fold_save_dir, "normalization_params.json")
        )

        # 当前折在整个 Optuna 搜索过程中只维护一个最优检查点。
        fold_checkpoint_state = {
            'path': None,
            'score': -np.inf,
            'trial_number': None,
        }

        # 创建Optuna研究
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed)
        )

        # 优化目标函数
        study.optimize(
            lambda trial: objective(
                trial,
                fold_train_mapping,
                fold_val_mapping,
                test_mapping,
                normalization_params,
                csv_tensor_cache_dir,
                fold_idx,
                fold_save_dir,
                fold_checkpoint_state,
            ),
            n_trials=30,
            show_progress_bar=True
        )

        # 保存最佳超参数
        best_params = dict(study.best_params)
        # Comment 11：类别权重不是待搜索超参数，而是当前折训练样本比例的确定值。
        best_params.update({
            'class_weight_negative': 1.0,
            'class_weight_positive': float(positive_class_weight),
            'train_negative_count': int(train_neg),
            'train_positive_count': int(train_pos),
            'class_weight_formula': 'N_negative / N_positive',
        })
        with open(os.path.join(fold_save_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)

        # 本折 Optuna 完成后，根目录中只应存在唯一的 best_model.ckpt。
        best_model_path = fold_checkpoint_state.get('path')
        best_f2 = float(fold_checkpoint_state.get('score', -np.inf))
        if best_model_path and os.path.exists(best_model_path):
            # 最后一次强制清理，确保本折没有其他 ckpt。
            delete_checkpoint_files(fold_save_dir, keep_path=best_model_path)
            remaining_ckpts = []
            for current_root, _, filenames in os.walk(fold_save_dir):
                remaining_ckpts.extend(
                    os.path.join(current_root, name)
                    for name in filenames
                    if name.endswith('.ckpt')
                )
            if len(remaining_ckpts) != 1:
                raise RuntimeError(
                    f"Fold {fold_idx} 应只保留 1 个 ckpt，实际为 {len(remaining_ckpts)}"
                )
            print(
                f"✅ Fold {fold_idx} 最终仅保留：{best_model_path} "
                f"(validation F2={best_f2:.6f}, "
                f"trial={fold_checkpoint_state.get('trial_number')})"
            )
            best_models.append(best_model_path)
        else:
            print(f"❌ No suitable model found for fold {fold_idx}")
            best_models.append(None)

    # 保存每折训练类别比例与实际损失权重。
    fold_training_summary_df = pd.DataFrame(fold_training_summaries)
    fold_training_summary_path = os.path.join(
        base_save_dir,
        'fold_training_class_ratio_and_weights.csv',
    )
    fold_training_summary_df.to_csv(fold_training_summary_path, index=False)

    # 使用所有折的最佳模型进行最终测试评估。
    # 每折模型使用与其配套的训练集归一化参数，并保存该折正类概率。
    final_results = []
    fold_test_probabilities = []
    reference_test_labels = None
    reference_test_paths = [file_path for file_path, _ in test_mapping]

    for fold_number, model_path in enumerate(best_models, start=1):
        if model_path is None:
            raise RuntimeError(
                f"第 {fold_number} 折没有找到最佳模型。"
                "五折集成和 Bootstrap 置信区间要求五折模型全部存在。"
            )

        print(f"\n🔍 评估第 {fold_number} 折的最佳模型在独立测试集上的表现")
        print(f"模型路径: {model_path}")

        model = RainfallModel.load_from_checkpoint(model_path)
        model.to(device)
        model.eval()
        param_count = model.count_parameters()

        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_number}")
        hyperparams_path = os.path.join(fold_save_dir, 'best_params.json')
        try:
            with open(hyperparams_path, 'r', encoding='utf-8') as f:
                hyperparams = json.load(f)
            hyperparams_str = json.dumps(hyperparams, ensure_ascii=False)
        except Exception as exc:
            print(f"❌ 无法读取超参数文件: {exc}")
            hyperparams_str = "{}"

        fold_normalization_path = os.path.join(
            fold_save_dir,
            'normalization_params.json'
        )
        fold_normalization_params = load_fold_normalization_params(
            fold_normalization_path
        )

        test_dataset = CSVDataset(
            test_mapping, fold_normalization_params, csv_tensor_cache_dir
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=40,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue

                inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key != 'labels'
                }
                labels = batch['labels'].to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        labels_array = np.asarray(all_labels, dtype=np.int64)
        probs_array = np.asarray(all_probs, dtype=np.float64)

        if len(labels_array) != len(test_mapping):
            raise RuntimeError(
                f"Fold {fold_number} 实际预测 {len(labels_array)} 个样本，"
                f"但 test_mapping 包含 {len(test_mapping)} 个样本。"
                "请检查是否有 CSV 读取失败。"
            )

        if reference_test_labels is None:
            reference_test_labels = labels_array.copy()
        elif not np.array_equal(reference_test_labels, labels_array):
            raise RuntimeError(
                f"Fold {fold_number} 的测试标签顺序与前一折不一致，"
                "不能直接平均概率。"
            )

        fold_test_probabilities.append(probs_array)
        fold_metrics = calculate_test_metrics(
            labels_array,
            probs_array,
            threshold=TEST_THRESHOLD,
        )

        result = {
            # Comment 5：显著性检验配对键。五次重复时只需修改顶部 seed；
            # PatchDC 与 PatchTST 必须使用相同 seed 和相同 fold。
            'model_name': MODEL_NAME,
            'run_id': EXPERIMENT_RUN_ID,
            'seed': seed,
            'fold': fold_number,
            'test_acc': fold_metrics['accuracy'],
            'test_precision_1': fold_metrics['precision_class_1'],
            'test_recall_1': fold_metrics['recall_class_1'],
            'test_f2': fold_metrics['f2_class_1'],
            'test_aupr': fold_metrics['aupr'],
            'param_count': param_count,
            'hyperparams': hyperparams_str,
            'model_path': model_path,
        }
        final_results.append(result)

        print(f"📊 第 {fold_number} 折测试结果:")
        print(f"准确率: {fold_metrics['accuracy']:.4f}")
        print(f"正类精确率: {fold_metrics['precision_class_1']:.4f}")
        print(f"正类召回率: {fold_metrics['recall_class_1']:.4f}")
        print(f"正类F2: {fold_metrics['f2_class_1']:.4f}")
        print(f"AUPR: {fold_metrics['aupr']:.4f}")
        print(f"参数量: {param_count}")

        fold_prediction_path = os.path.join(
            base_save_dir,
            f'test_probabilities_fold_{fold_number}.csv'
        )
        pd.DataFrame({
            'file_path': reference_test_paths,
            'true_label': labels_array,
            'probability_class_1': probs_array,
            'prediction_at_0.5': (probs_array >= TEST_THRESHOLD).astype(int),
        }).to_csv(fold_prediction_path, index=False)

    # 保存五个单折模型的测试指标。
    results_df = pd.DataFrame(final_results)
    results_path = os.path.join(base_save_dir, 'final_results_per_fold.csv')
    results_df.to_csv(results_path, index=False)

    # 同时保存一份只包含显著性检验所需字段的标准化表。
    # 五个 seed 分别运行后，将五份该文件与 PatchTST 的五份对应文件交给
    # paired_ttest_wilcoxon.py，即可得到 25 对配对 t 检验和 Wilcoxon 检验。
    significance_ready_path = os.path.join(
        base_save_dir,
        'significance_ready_metrics.csv'
    )
    results_df[[
        'model_name', 'run_id', 'seed', 'fold',
        'test_acc', 'test_precision_1', 'test_recall_1',
        'test_f2', 'test_aupr'
    ]].to_csv(significance_ready_path, index=False)

    if len(fold_test_probabilities) != n_splits:
        raise RuntimeError(
            f"只有 {len(fold_test_probabilities)} 折概率，预期为 {n_splits} 折。"
        )

    # -------------------------------------------------------------------------
    # 五折概率集成：这是论文最终独立测试结果的计算对象。
    # -------------------------------------------------------------------------
    fold_probability_matrix = np.stack(fold_test_probabilities, axis=0)
    ensemble_probability = np.mean(fold_probability_matrix, axis=0)
    ensemble_prediction = (
        ensemble_probability >= TEST_THRESHOLD
    ).astype(np.int64)

    ensemble_metrics = calculate_test_metrics(
        reference_test_labels,
        ensemble_probability,
        threshold=TEST_THRESHOLD,
    )

    # Comment 10：一票赞成（one-vote positive）策略。
    # 只要五个折模型中任意一个在阈值 0.5 下判断为正类，最终即为正类。
    # max_probability 在阈值 0.5 下与上述 OR 规则完全等价，同时可用于计算 AUPR。
    one_vote_probability = np.max(fold_probability_matrix, axis=0)
    positive_vote_count = np.sum(
        fold_probability_matrix >= TEST_THRESHOLD,
        axis=0,
    ).astype(np.int64)
    one_vote_prediction = (positive_vote_count >= 1).astype(np.int64)
    if not np.array_equal(
        one_vote_prediction,
        (one_vote_probability >= TEST_THRESHOLD).astype(np.int64),
    ):
        raise RuntimeError("one-vote OR 结果与最大概率阈值结果不一致")

    one_vote_metrics = calculate_test_metrics(
        reference_test_labels,
        one_vote_probability,
        threshold=TEST_THRESHOLD,
    )

    ensemble_predictions = pd.DataFrame({
        'file_path': reference_test_paths,
        'true_label': reference_test_labels,
        'mean_probability_class_1': ensemble_probability,
        'prediction_at_0.5': ensemble_prediction,
        'max_probability_class_1': one_vote_probability,
        'positive_vote_count_at_0.5': positive_vote_count,
        'one_vote_positive_at_0.5': one_vote_prediction,
    })
    for fold_number in range(1, n_splits + 1):
        ensemble_predictions[
            f'fold_{fold_number}_probability_class_1'
        ] = fold_probability_matrix[fold_number - 1]

    ensemble_prediction_path = os.path.join(
        base_save_dir,
        'ensemble_test_predictions.csv'
    )
    ensemble_predictions.to_csv(ensemble_prediction_path, index=False)

    # Comment 10：保存“平均概率集成”和“一票赞成”两种策略的并列结果。
    strategy_comparison_df = pd.DataFrame([
        build_strategy_result_row(
            'mean_probability_ensemble',
            reference_test_labels,
            ensemble_probability,
            threshold=TEST_THRESHOLD,
        ),
        build_strategy_result_row(
            'one_vote_positive',
            reference_test_labels,
            one_vote_probability,
            threshold=TEST_THRESHOLD,
        ),
    ])
    strategy_comparison_path = os.path.join(
        base_save_dir,
        'ensemble_strategy_comparison.csv',
    )
    strategy_comparison_df.to_csv(strategy_comparison_path, index=False)

    one_vote_prediction_path = os.path.join(
        base_save_dir,
        'one_vote_positive_test_predictions.csv',
    )
    ensemble_predictions[[
        'file_path',
        'true_label',
        'max_probability_class_1',
        'positive_vote_count_at_0.5',
        'one_vote_positive_at_0.5',
    ]].to_csv(one_vote_prediction_path, index=False)

    # Comment 9：保存主结果的 PR 曲线、曲线坐标以及 0.5/0.4 工作点。
    pr_curve_outputs = save_precision_recall_curve(
        reference_test_labels,
        ensemble_probability,
        base_save_dir,
        file_prefix='ensemble_precision_recall_curve',
        highlighted_thresholds=(0.5, 0.4),
    )

    # -------------------------------------------------------------------------
    # Comment 4：在最终五折集成预测上进行 10,000 次分层 Bootstrap。
    # 这反映测试样本有限造成的抽样不确定性，不替代多次训练的均值±标准差。
    # -------------------------------------------------------------------------
    (
        ensemble_point_estimates,
        bootstrap_summary_df,
        bootstrap_distribution_df,
    ) = stratified_bootstrap_confidence_intervals(
        reference_test_labels,
        ensemble_probability,
        threshold=TEST_THRESHOLD,
        n_resamples=BOOTSTRAP_N_RESAMPLES,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        random_seed=BOOTSTRAP_RANDOM_SEED,
    )

    bootstrap_summary_path = os.path.join(
        base_save_dir,
        'ensemble_metrics_bootstrap_95ci.csv'
    )
    bootstrap_summary_df.to_csv(bootstrap_summary_path, index=False)

    bootstrap_distribution_path = os.path.join(
        base_save_dir,
        'ensemble_bootstrap_metric_distribution.csv.gz'
    )
    bootstrap_distribution_df.to_csv(
        bootstrap_distribution_path,
        index=False,
        compression='gzip',
    )

    bootstrap_json_path = os.path.join(
        base_save_dir,
        'ensemble_metrics_bootstrap_95ci.json'
    )
    bootstrap_json = {
        'test_sample_count': int(len(reference_test_labels)),
        'positive_test_sample_count': int(np.sum(reference_test_labels == 1)),
        'negative_test_sample_count': int(np.sum(reference_test_labels == 0)),
        'threshold': TEST_THRESHOLD,
        'n_fold_models': n_splits,
        'bootstrap_method': 'stratified nonparametric percentile bootstrap',
        'bootstrap_n_resamples': BOOTSTRAP_N_RESAMPLES,
        'confidence_level': BOOTSTRAP_CONFIDENCE_LEVEL,
        'random_seed': BOOTSTRAP_RANDOM_SEED,
        'one_vote_positive': {
            'definition': (
                'final positive if at least one of the five fold models has '
                'probability >= 0.5'
            ),
            'accuracy': one_vote_metrics['accuracy'],
            'precision_class_1': one_vote_metrics['precision_class_1'],
            'recall_class_1': one_vote_metrics['recall_class_1'],
            'f2_class_1': one_vote_metrics['f2_class_1'],
            'aupr_from_max_probability': one_vote_metrics['aupr'],
        },
        'precision_recall_curve': {
            'png_path': pr_curve_outputs['png_path'],
            'pdf_path': pr_curve_outputs['pdf_path'],
            'curve_csv_path': pr_curve_outputs['curve_csv_path'],
            'operating_csv_path': pr_curve_outputs['operating_csv_path'],
            'aupr': pr_curve_outputs['aupr'],
            'positive_prevalence': pr_curve_outputs['positive_prevalence'],
        },
        'metrics': {
            row['metric']: {
                'estimate': float(row['estimate']),
                'ci_lower': float(row['ci_lower']),
                'ci_upper': float(row['ci_upper']),
            }
            for _, row in bootstrap_summary_df.iterrows()
        },
    }
    with open(bootstrap_json_path, 'w', encoding='utf-8') as f:
        json.dump(bootstrap_json, f, ensure_ascii=False, indent=2)

    # 测试结束后生成简短汇总报告，集中列出所有关键结果与文件位置。
    report_output_files = {
        '每折测试指标': results_path,
        '每折训练类别比例与权重': fold_training_summary_path,
        '显著性检验标准表': significance_ready_path,
        '五折集成逐样本预测': ensemble_prediction_path,
        '平均概率与一票赞成策略比较': strategy_comparison_path,
        '一票赞成逐样本预测': one_vote_prediction_path,
        'PR曲线PNG': pr_curve_outputs['png_path'],
        'PR曲线PDF': pr_curve_outputs['pdf_path'],
        'PR曲线坐标': pr_curve_outputs['curve_csv_path'],
        'PR曲线工作点': pr_curve_outputs['operating_csv_path'],
        'Bootstrap置信区间CSV': bootstrap_summary_path,
        'Bootstrap置信区间JSON': bootstrap_json_path,
        'Bootstrap完整分布': bootstrap_distribution_path,
    }
    report_paths = write_brief_test_report(
        output_dir=base_save_dir,
        per_fold_results_df=results_df,
        fold_training_summary_df=fold_training_summary_df,
        strategy_comparison_df=strategy_comparison_df,
        bootstrap_summary_df=bootstrap_summary_df,
        pr_curve_outputs=pr_curve_outputs,
        retained_checkpoint_paths=best_models,
        test_labels=reference_test_labels,
        threshold=TEST_THRESHOLD,
        output_files=report_output_files,
    )

    completion_path = os.path.join(base_save_dir, 'RUN_COMPLETED.json')
    completion_payload = {
        'model_name': MODEL_NAME,
        'seed': seed,
        'run_id': EXPERIMENT_RUN_ID,
        'completed_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'brief_report_markdown': report_paths['markdown_path'],
        'brief_report_text': report_paths['text_path'],
        'status': 'completed',
    }
    with open(completion_path, 'w', encoding='utf-8') as f:
        json.dump(completion_payload, f, ensure_ascii=False, indent=2)

    print("\n🎉 五折训练、独立测试集成与 Bootstrap 置信区间计算完成")
    print(f"单折结果: {results_path}")
    print(f"显著性检验标准表: {significance_ready_path}")
    print(f"集成预测: {ensemble_prediction_path}")
    print(f"两种集成策略结果: {strategy_comparison_path}")
    print(f"一票赞成逐样本结果: {one_vote_prediction_path}")
    print(f"PR 曲线 PNG: {pr_curve_outputs['png_path']}")
    print(f"PR 曲线 PDF: {pr_curve_outputs['pdf_path']}")
    print(f"PR 曲线数据: {pr_curve_outputs['curve_csv_path']}")
    print(f"Bootstrap 95% CI: {bootstrap_summary_path}")
    print(f"测试简报 Markdown: {report_paths['markdown_path']}")
    print(f"测试简报 TXT: {report_paths['text_path']}")

    print("\n一票赞成策略（任一折概率 >= 0.5 即为正类）:")
    print(f"  Accuracy: {one_vote_metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {one_vote_metrics['precision_class_1'] * 100:.2f}%")
    print(f"  Recall: {one_vote_metrics['recall_class_1'] * 100:.2f}%")
    print(f"  F2: {one_vote_metrics['f2_class_1'] * 100:.2f}%")
    print(f"  AUPR（以五折最大概率为连续评分）: {one_vote_metrics['aupr'] * 100:.2f}%")

    print("\n最终五折平均概率集成指标及 95% Bootstrap CI:")
    for _, row in bootstrap_summary_df.iterrows():
        print(
            f"  {row['metric']}: {row['estimate_percent']:.2f}% "
            f"[{row['ci_lower_percent']:.2f}%, "
            f"{row['ci_upper_percent']:.2f}%]"
        )
