"""生成合成示例数据集到 outputs/sample_dataset/（CLI 入口）。

用法::

    python scripts/make_sample_dataset.py [--out 目标目录]

数据不进 git（.gitignore 已排除 outputs/）；生成是确定性的（固定种子），
两次运行内容逐字节一致。UI 侧栏"数据加载"面板中的"加载示例数据集"按钮
会自动调用同一生成逻辑（app/ui/sample_dataset.py），无需先跑本脚本。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 使脚本可直接运行（无包安装）时也能 import app 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    """解析参数并生成示例数据集，打印各文件行数。"""
    parser = argparse.ArgumentParser(description="生成合成示例数据集")
    parser.add_argument("--out", default=None, help="目标目录（缺省 outputs/sample_dataset）")
    args = parser.parse_args()

    from app.ui.sample_dataset import ensure_sample_dataset, generate_sample_frames

    d = ensure_sample_dataset(args.out) if args.out else ensure_sample_dataset()
    if args.out:  # ensure_sample_dataset 对显式目录同样处理缺文件生成
        pass
    print(f"示例数据集已就绪：{d}")
    for name, df in generate_sample_frames().items():
        print(f"  {name}: {len(df)} 行")


if __name__ == "__main__":
    main()
