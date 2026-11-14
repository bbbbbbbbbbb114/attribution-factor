# RQ1 第一轮先导实验

采用 GEPA，条件为 Baseline、1.1 Causing、1.2 Inevitable。任务模型为 `qwen3.6-35b-a3b`，优化与预标注模型为 `deepseek-v4.1-flash-1017`。模型角色为本轮准备时的暂定选择。

## 输入与划分

复制 `rq1-pilot.example.json` 到本地数据目录，填写实际路径与任务 ID。相对数据路径按配置文件所在目录解析。三个任务列表必须非空、无重复、互斥。`pilot_eval` 用于方案开发与先导分析。其余正式测试任务继续封存。

训练样本需要 schema 2.0 的失败分析 JSON，包含两个锚点及 `trace.path`。相对轨迹路径沿用仓库约定，按仓库根目录解析。dev 和 pilot_eval 只需要任务数据。公开 SWE-bench parquet 可以下载，原始运行轨迹与适用于本 harness 的锚点需要收集或提供。

三组均向 GEPA 反思阶段提供相同的原始训练轨迹消息。处理组额外提供相应位置与机械提取的原始片段。准备过程只读取训练标注，生成三个条件，并记录同点比例、距离、末步比例、片段字符数、截断情况及审核状态。

`bootstrap_pilot` 支持从预先声明的训练候选收集 250 步上限的原始轨迹，对经过官方评测的失败生成 DeepSeek 预标注。预标注的 `review_status` 为 `llm_preannotation_pending_independent_review`。位置校验可以验证索引与消息一致，语义审核另行记录。模型返回不确定标签时保留原结果供审核。

```bash
python -m attribution_experiments.bootstrap_pilot \
  --selection /volume/med/experiment/data/bootstrap-selection.json \
  --parquet /volume/med/experiment/data/tasks.parquet \
  --reference-trace /volume/med/experiment/data/reference.traj.json \
  --output /volume/med/experiment/data/bootstrap \
  --task-id REAL_TRAINING_TASK_ID

python -m attribution_experiments.pilot_inputs \
  --config /volume/med/experiment/data/pilot.json \
  --output /volume/med/experiment/data/frozen-rq1-v1
```

准备后的 bundle 包含 manifest、完整训练输入和训练 ID 列表。校验和覆盖 parquet、参考 harness、训练分析、训练轨迹和处理包。重新准备使用新目录。实际任务缺失或重复会在读取 parquet 时拒绝运行。

## 执行环境

使用 Linux、Python 3.12 与 Docker。先安装固定依赖与仓库中的两个包。

```bash
python -m pip install -r attribution_experiments/requirements-pilot.txt
python -m pip install -e baselines/gepa -e agent_runtime/mini-swe-agent-v2.4.6
export PYTHONDONTWRITEBYTECODE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export EXPERIMENT_CONTAINER_BACKEND=local_docker
export UDOCKER_RUNTIME_ROOT=/volume/med/experiment/.codex-tmp/runtime/containers
export TMPDIR=/volume/med/experiment/.codex-tmp
export TASK_MODEL_BASE_URL=YOUR_OPENAI_COMPATIBLE_ENDPOINT
export OPTIMIZER_BASE_URL=YOUR_OPENAI_COMPATIBLE_ENDPOINT
```

将 `TASK_MODEL_API_KEY` 和 `OPTIMIZER_API_KEY` 注入进程环境。也可用进程中的 `ALIYUN_API_KEY` 作为共用凭据。端点和密钥放在本地环境配置中。

请求参数随 bundle 冻结。示例将任务输出上限设为 4096 token，将 DeepSeek 的 `enable_thinking` 设为 false。原始标注响应与 token 用量保存在 `annotation_responses/`，便于解析失败后复核。遇到服务配额错误时按基础设施失败处理。

初步实验使用 `rq1-throughput.env` 中的吞吐配置。Qwen 同时在途请求上限为 4，滚动 60 秒 token 预算为 1,000,000，使用比例为 100%，请求最小间隔为 0.1 秒。每个请求先预留 131,072 token，响应后按实际用量结算。任务执行与评测的 worker 数均为 3，当前 WSL 约 7.4 GiB 内存，每个容器上限为 2 GiB。

```bash
. attribution_experiments/rq1-throughput.env
```

`rq1_pilot` 将 `GEPA_EVOLUTION_WORKERS` 和 `GEPA_FINAL_WORKERS` 传入原生 GEPA，命令行也支持 `--evolution-workers`、`--final-workers`。预检查输出实际 worker 数。每个运行会记录该并发配置和请求门控参数，续跑时校验一致性。候选依赖前轮结果，优化提案按 GEPA 原生顺序推进。

新增 `local_docker` 后端可直接使用已经导入的官方镜像。它按镜像摘要启动容器，沿用共享 Agent 的动作、观察和提交处理，通过 stdin 传递补丁与评测脚本。容器停止后保留文件与镜像供检查。用户自行安排保留资源的清理。

## 检查与运行

```bash
python -m attribution_experiments.rq1_pilot \
  --bundle /volume/med/experiment/data/frozen-rq1-v1
```

检查覆盖本地输入、依赖、模型配置与存储路径。成功输出的 `checks` 字段标明检查范围。Docker 服务、缓存镜像和完整官方评测还需要通过实际冒烟运行验证。

每个 seed 先独立重跑 H0，再运行三组。以下命令循环使用冻结预算中的 seed。

```bash
for seed in 42 43 44; do
  python -m attribution_experiments.rq1_pilot \
    --bundle /volume/med/experiment/data/frozen-rq1-v1 \
    --seed "$seed" --initial-only --run
  for group in baseline group1_1_causing group1_2_inevitable; do
    python -m attribution_experiments.rq1_pilot \
      --bundle /volume/med/experiment/data/frozen-rq1-v1 \
      --seed "$seed" --group "$group" --run
  done
done
```

每组每个 seed 使用独立目录与引擎状态。候选评估和晋升使用 train/dev，选定 scaffold 后在 pilot_eval 上冻结评测，并在 `historical_recovery/` 单独评估训练失败任务。H0 直接评测原 scaffold。候选未改善时保留原 scaffold 并计入结果。

输出位于 `attribution_experiments/gepa/<group>/rq1_<bundle>_<group或h0>_seed<seed>/`。`evolution_summary.json` 保存训练、选择和评测 ID、实际候选数、seed 与本进程的优化模型 token 用量。逐调用目录保留真实轨迹、补丁和官方评测报告。自定义模型的美元成本需要结合服务商实际价格计算。

## 暂定规模与预算

示例采用 4 个候选提案、每次反思 3 个训练样本、任务 250 步上限和 3 个优化 seed。三组共同的主约束为候选评估次数与任务执行上限。输入长度会影响 token 消耗，当前入口记录实际用量。共同优化 token 硬上限与端到端归因成本匹配留待预算确定后实现。

初始数据下载与 3 步冒烟运行用于管线检查。正式训练集应来自真实失败轨迹，并在实验 manifest 中报告实际审核覆盖。先导评测若用于调整方案，应继续归入开发过程。
