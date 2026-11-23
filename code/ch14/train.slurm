#!/bin/bash
# クラスタではこう書く（このままは実行していない）。
# 2 ノード × 8 GPU = 16 プロセスで ch14/ddp_train.py を動かす Slurm のジョブ。
#SBATCH --job-name=minigpt-ddp
#SBATCH --nodes=2                 # ノード（マシン）の数
#SBATCH --ntasks-per-node=1       # 各ノードで torchrun を 1 つ起動
#SBATCH --gpus-per-node=8         # 各ノードの GPU 8 枚を確保
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00           # 最長の実行時間（過ぎると強制終了）
#SBATCH --output=logs/%x-%j.out   # %x=ジョブ名, %j=ジョブ番号

# 先頭ノードを集合場所（rendezvous）にする
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=8 \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head:29500" \
  ch14/ddp_train.py
