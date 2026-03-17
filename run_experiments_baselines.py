"""
run_experiments_baselines.py
============================
自动化批量实验运行脚本，参考 run_experiments_fedavg.py 的设计：
  - 参数网格搜索 (Parameter grid search)
  - 断点续传 (Checkpoint/resume)
  - 进度日志 (Progress logging)
  - 失败跳过 (Failure skip with logging)

所有输出统一保存在 outputs/ 目录下。
All outputs are saved under the outputs/ directory.

Automated batch experiment runner for baseline methods.
Supports checkpoint/resume, progress logging, and failure handling.
"""

import subprocess
import itertools
import json
import os
import sys
from datetime import datetime


# ============================================================
# 参数配置区域 —— 按需修改 (Parameter Configuration — Modify as needed)
# ============================================================

param_grid = {
    # --- 选择要跑的 baseline 方法 ---
    # --- Baseline methods to run ---
    # 支持以下方法："iafl", "cgsv", "fedavgft", "lgfedavg", "rank"
    "method":       ["iafl", "cgsv", "rank", "fedavgft", "lgfedavg"],

    # --- 数据集（SST 已移除：torchtext 不兼容 PyTorch >= 2.2） ---
    # --- Datasets (SST removed: torchtext incompatible with PyTorch >= 2.2) ---
    "dataset":      ["mnist", "fashion-mnist", "cifar10"],

    # --- 数据分布（全部四种场景） ---
    "distribution": ["iid", "non-iid-dir", "non-iid-size", "non-iid-class"],

    # --- 分布专属参数（仅在对应分布下生效，其余场景由过滤函数自动跳过） ---
    "alpha":                  [0.5],        # 仅 non-iid-dir 使用
    "size_imbalance_ratio":   [5.0],        # 仅 non-iid-size 使用
    "min_classes_per_client":  [2],          # 仅 non-iid-class 使用
    "max_classes_per_client":  [5],          # 仅 non-iid-class 使用

    # --- 客户端数量 ---
    "num_clients":  [5, 10, 15, 20, 25, 30],

    # --- 训练参数 ---
    # 注意: 原始论文中 MNIST/FMNIST=50, CIFAR10=100, CIFAR100=100(CNN)/1000(ResNet)
    # 请根据数据集手动调整此值，20 轮对 CIFAR 系列严重不足
    # Note: original paper uses MNIST/FMNIST=50, CIFAR10=100, CIFAR100=100(CNN)/1000(ResNet)
    # Adjust manually per dataset; 20 rounds is severely insufficient for CIFAR
    "num_rounds":   [50],
    "local_epochs": [1],
    "standalone_epochs": [50],

    # --- 其他 ---
    "seed":         [42],
    "gpu":          [0],
    "model":        ["cnn"],
}

# 输出根目录 (Output root directory)
OUTPUT_DIR = "outputs"

# 断点文件路径 (Checkpoint file path)
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "run_experiments_baselines_checkpoint.json")

# 汇总报告文件 (Summary report file)
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "experiments_summary.json")

# 单个实验最大超时时间（秒）(Max timeout per experiment in seconds)
TIMEOUT_SECONDS = 7200  # 2 小时

# ============================================================


def load_checkpoint():
    """读取断点：返回已完成的实验索引集合和失败列表"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        print(f"[Resume] Found checkpoint: {len(data['completed'])} experiments done, "
              f"{len(data['failed'])} failed.\n")
        return set(data["completed"]), data["failed"]
    return set(), []


def save_checkpoint(completed: set, failed: list):
    """保存断点"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "completed": list(completed),
            "failed": failed,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)


def filter_valid_combinations(combinations, keys):
    """
    过滤无效的参数组合：
    每种分布只关心自己的专属参数，其他分布的专属参数必须是默认值，
    否则会产生笛卡尔积中的冗余组合。

    规则 (Rules):
      - iid          : alpha=0.5, ratio=5.0, min=2, max=5 (全默认)
      - non-iid-dir  : alpha 可变; ratio=5.0, min=2, max=5
      - non-iid-size : ratio 可变; alpha=0.5, min=2, max=5
      - non-iid-class: min/max 可变; alpha=0.5, ratio=5.0

    Filter out invalid parameter combinations to avoid cartesian product redundancy.
    """
    # 各参数的默认值 (Default values)
    DEFAULTS = {
        'alpha': 0.5,
        'size_imbalance_ratio': 5.0,
        'min_classes_per_client': 2,
        'max_classes_per_client': 5,
    }

    valid = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        dist = params.get('distribution', '')

        skip = False

        if dist == 'iid':
            # iid: 所有专属参数必须为默认值
            for k, default in DEFAULTS.items():
                if params.get(k, default) != default:
                    skip = True
                    break

        elif dist == 'non-iid-dir':
            # non-iid-dir: alpha 可变，其余必须默认
            for k in ['size_imbalance_ratio', 'min_classes_per_client', 'max_classes_per_client']:
                if params.get(k, DEFAULTS[k]) != DEFAULTS[k]:
                    skip = True
                    break

        elif dist == 'non-iid-size':
            # non-iid-size: size_imbalance_ratio 可变，其余必须默认
            for k in ['alpha', 'min_classes_per_client', 'max_classes_per_client']:
                if params.get(k, DEFAULTS[k]) != DEFAULTS[k]:
                    skip = True
                    break

        elif dist == 'non-iid-class':
            # non-iid-class: min/max_classes 可变，其余必须默认
            for k in ['alpha', 'size_imbalance_ratio']:
                if params.get(k, DEFAULTS[k]) != DEFAULTS[k]:
                    skip = True
                    break

        if not skip:
            valid.append(combo)

    return valid


def build_command(params):
    """
    构建 main_baseline_unified.py 的命令行参数
    Build command line arguments for main_baseline_unified.py
    """
    cmd = [sys.executable, "main_baseline_unified.py"]

    # 参数映射 (Parameter mapping)
    for k, v in params.items():
        cmd.extend([f"--{k}", str(v)])

    # 添加输出目录参数
    cmd.extend(["--output_dir", OUTPUT_DIR])

    return cmd


def collect_metrics(output_dir, method, params):
    """
    从已完成实验的输出目录中收集指标
    Collect metrics from completed experiment's output directory
    """
    exp_name = (f"{method}_{params['dataset']}_{params['distribution']}_"
                f"N{params['num_clients']}_R{params['num_rounds']}_seed{params['seed']}")
    metrics_path = os.path.join(output_dir, method, exp_name, 'metrics.json')

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def save_summary(all_metrics):
    """
    保存所有实验的汇总报告
    Save summary report of all experiments
    """
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, 'w') as f:
        json.dump({
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_experiments": len(all_metrics),
            "experiments": all_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[Summary] Saved to: {SUMMARY_FILE}")


def main():
    print("=" * 70)
    print("  Baseline Experiments — Automated Batch Runner")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 生成所有参数组合
    keys = list(param_grid.keys())
    all_combinations = list(itertools.product(*param_grid.values()))

    # 过滤无效组合
    valid_combinations = filter_valid_combinations(all_combinations, keys)
    total = len(valid_combinations)

    print(f"\nTotal experiments (after filtering): {total}")
    print(f"Parameter grid: {json.dumps({k: v for k, v in param_grid.items()}, indent=2)}")

    # 加载断点
    completed, failed = load_checkpoint()
    skipped = len(completed)
    if skipped > 0:
        print(f"\nSkipping {skipped} already-completed experiments.\n")

    # 收集所有已完成实验的指标
    all_metrics = []

    for i, combo in enumerate(valid_combinations):
        # 已完成的直接跳过
        if i in completed:
            params = dict(zip(keys, combo))
            metrics = collect_metrics(OUTPUT_DIR, params['method'], params)
            if metrics:
                all_metrics.append(metrics)
            continue

        params = dict(zip(keys, combo))
        cmd = build_command(params)

        exp_desc = (f"{params['method']} | {params['dataset']} | "
                    f"{params['distribution']} | N={params['num_clients']} | "
                    f"R={params['num_rounds']}")

        print(f"\n{'─'*70}")
        print(f"[{i+1}/{total}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Experiment: {exp_desc}")
        print(f"  CMD: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                timeout=TIMEOUT_SECONDS,
                capture_output=False,  # 让输出直接打印到终端
            )

            if result.returncode == 0:
                completed.add(i)
                save_checkpoint(completed, failed)
                print(f"  ✓ Success\n")

                # 收集指标
                metrics = collect_metrics(OUTPUT_DIR, params['method'], params)
                if metrics:
                    all_metrics.append(metrics)
                    print(f"    Accuracy={metrics.get('global_accuracy', 'N/A'):.4f}, "
                          f"PCC={metrics.get('PCC', 'N/A'):.4f}, "
                          f"IPR={metrics.get('IPR', 'N/A'):.4f}")
            else:
                reason = f"returncode={result.returncode}"
                failed.append({
                    "index": i,
                    "cmd": ' '.join(cmd),
                    "params": params,
                    "reason": reason,
                    "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
                save_checkpoint(completed, failed)
                print(f"  ✗ Failed ({reason}), skipping.\n")

        except subprocess.TimeoutExpired:
            reason = f"timeout (>{TIMEOUT_SECONDS}s)"
            failed.append({
                "index": i,
                "cmd": ' '.join(cmd),
                "params": params,
                "reason": reason,
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
            save_checkpoint(completed, failed)
            print(f"  ✗ Failed ({reason}), skipping.\n")

        except KeyboardInterrupt:
            save_checkpoint(completed, failed)
            save_summary(all_metrics)
            print("\n[Interrupted] Progress saved. Re-run to resume.")
            return

        except Exception as e:
            reason = str(e)
            failed.append({
                "index": i,
                "cmd": ' '.join(cmd),
                "params": params,
                "reason": reason,
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
            save_checkpoint(completed, failed)
            print(f"  ✗ Unexpected error: {reason}, skipping.\n")

    # ========== 全部完成，生成汇总 ==========
    print("\n" + "=" * 70)
    print(f"All experiments finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Completed : {len(completed)}/{total}")
    print(f"  Failed    : {len(failed)}/{total}")

    if failed:
        print("\nFailed experiments:")
        for item in failed:
            print(f"  [{item['index']+1}] {item.get('reason', 'unknown')}")
            if 'params' in item:
                p = item['params']
                print(f"       {p.get('method','')} | {p.get('dataset','')} | "
                      f"{p.get('distribution','')} | N={p.get('num_clients','')}")

    # 保存汇总报告
    save_summary(all_metrics)

    # 打印指标汇总表
    if all_metrics:
        print("\n" + "=" * 70)
        print("  Metrics Summary")
        print("=" * 70)
        print(f"{'Method':<12} {'Dataset':<14} {'Distribution':<14} "
              f"{'N':>3} {'R':>4} {'Accuracy':>9} {'PCC':>7} {'IPR':>7}")
        print("-" * 75)
        for m in all_metrics:
            print(f"{m.get('method',''):<12} "
                  f"{m.get('dataset',''):<14} "
                  f"{m.get('distribution',''):<14} "
                  f"{m.get('num_clients',''):>3} "
                  f"{m.get('num_rounds',''):>4} "
                  f"{m.get('global_accuracy',0):>9.4f} "
                  f"{m.get('PCC',0):>7.4f} "
                  f"{m.get('IPR',0):>7.4f}")

    # 清理断点文件
    if len(completed) + len(failed) == total:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("\nCheckpoint file removed (all experiments completed).")


if __name__ == "__main__":
    main()