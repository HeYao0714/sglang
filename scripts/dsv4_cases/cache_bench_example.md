# Cache benchmark example

脚本内已固定：

- GSM8K数据源：`/home/t00937989/datasets/gsm8k/test.jsonl`
- Tokenizer：`/home/weights/DeepSeek-V4-Pro-w4a8-mtp`
- Page size：`128`
- 数据集根目录：`/home/t00937989/datasets/gsm8k`

## 生成64K输入、1K输出、90% Cache数据

```bash
cd /home/t00937989/sglang/scripts/dsv4_cases
chmod +x build_gsm8k_bench_dataset.sh bench.sh

./build_gsm8k_bench_dataset.sh \
  --target-tokens 64000 \
  --output-tokens 1000 \
  --cache-ratio 0.90 \
  --warm-prompts 2 \
  --formal-prompts 4 \
  --num-runs 3
```

生成目录：

```text
/home/t00937989/datasets/gsm8k/cache90_64000
```

## 检查bench读取的参数

```bash
DRY_RUN=1 ./bench.sh \
  /home/t00937989/datasets/gsm8k/cache90_64000
```

## 正式压测

```bash
HOST=127.0.0.1 PORT=31000 ./bench.sh \
  /home/t00937989/datasets/gsm8k/cache90_64000
```

`bench.sh`会从 `manifest.json` 自动读取输出长度、预热请求数、正式并发、测试轮数和每轮数据文件。脚本只预热一次，不会清理Cache，三轮正式测试使用不同后缀的数据。

## 查看Cache报告

```bash
grep -Ei 'cache|cached' \
  /home/t00937989/outputs/high_throughput_bench/*formal_run*.log
```

其他case只需要修改 `--target-tokens`、`--output-tokens`、`--cache-ratio`、请求数和轮数。公共前缀会按照128-token page向下对齐，实际比例记录在生成目录的 `manifest.json` 中。
