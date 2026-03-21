"""示例：使用 sageLLM Benchmark 对标 vLLM/SGLang 吞吐量测试。

本示例展示如何：
1. 运行 batch 模式（对标 vLLM offline throughput）
2. 运行 traffic 模式（对标 SGLang serving benchmark）
3. 提取和比较吞吐量指标

对标参考：
- vLLM: benchmarks/throughput.py
- SGLang: bench_offline_throughput.py
"""

import asyncio
import json
from pathlib import Path

from sagellm_core import LLMEngine, LLMEngineConfig

from sagellm_benchmark.runner import BenchmarkConfig, BenchmarkRunner
from sagellm_benchmark.workloads import TPCH_WORKLOADS


async def run_batch_mode_example():
    """运行 Batch 模式示例（对标 vLLM offline throughput）。"""
    print("=" * 80)
    print("示例 1: Batch 模式（Offline Throughput）")
    print("对标：vLLM benchmarks/throughput.py")
    print("=" * 80)

    # 创建引擎
    config = LLMEngineConfig(
        model_path="sshleifer/tiny-gpt2",
        backend_type="cpu",
        comm_type="gloo",
        max_batch_size=32,
        max_model_len=1024,
        max_new_tokens=128,
    )

    engine = LLMEngine(config)
    await engine.start()

    # 配置 benchmark（batch 模式）
    bench_config = BenchmarkConfig(
        engine=engine,
        workloads=TPCH_WORKLOADS,
        output_dir=Path("./outputs/batch_mode_example"),
        verbose=True,
        mode="batch",  # Batch 模式
    )

    runner = BenchmarkRunner(bench_config)
    results = await runner.run()

    # 输出关键指标
    print("\n" + "=" * 80)
    print("Batch 模式结果（vLLM/SGLang 兼容格式）")
    print("=" * 80)

    for workload_name, metrics in results.items():
        print(f"\n{workload_name}:")
        print(f"  Request Throughput:  {metrics.request_throughput_rps:>8.2f} req/s")
        print(f"  Input Throughput:    {metrics.input_throughput_tps:>8.2f} tokens/s")
        print(f"  Output Throughput:   {metrics.output_throughput_tps:>8.2f} tokens/s")
        print(f"  Total Throughput:    {metrics.total_throughput_tps:>8.2f} tokens/s")
        print(f"  Total Input Tokens:  {metrics.total_input_tokens:>8d}")
        print(f"  Total Output Tokens: {metrics.total_output_tokens:>8d}")
        print(f"  Total Time:          {metrics.total_time_s:>8.2f} s")

    await engine.stop()


async def run_traffic_mode_example():
    """运行 Traffic 模式示例（对标 SGLang serving benchmark）。"""
    print("\n\n" + "=" * 80)
    print("示例 2: Traffic 模式（Arrival Pattern Simulation）")
    print("对标：SGLang bench_serving.py")
    print("=" * 80)

    # 创建引擎
    config = LLMEngineConfig(
        model_path="sshleifer/tiny-gpt2",
        backend_type="cpu",
        comm_type="gloo",
        max_batch_size=32,
        max_model_len=1024,
        max_new_tokens=128,
    )

    engine = LLMEngine(config)
    await engine.start()

    # 配置 benchmark（traffic 模式）
    bench_config = BenchmarkConfig(
        engine=engine,
        workloads=TPCH_WORKLOADS,
        output_dir=Path("./outputs/traffic_mode_example"),
        verbose=True,
        mode="traffic",  # Traffic 模式
    )

    runner = BenchmarkRunner(bench_config)
    results = await runner.run()

    # 输出关键指标
    print("\n" + "=" * 80)
    print("Traffic 模式结果（vLLM/SGLang 兼容格式）")
    print("=" * 80)

    for workload_name, metrics in results.items():
        print(f"\n{workload_name}:")
        print(f"  Request Throughput:  {metrics.request_throughput_rps:>8.2f} req/s")
        print(f"  Input Throughput:    {metrics.input_throughput_tps:>8.2f} tokens/s")
        print(f"  Output Throughput:   {metrics.output_throughput_tps:>8.2f} tokens/s")
        print(f"  Total Throughput:    {metrics.total_throughput_tps:>8.2f} tokens/s")
        print(f"  Avg TTFT (P95):      {metrics.p95_ttft_ms:>8.2f} ms")
        print(f"  Avg TPOT (P95):      {metrics.p95_tpot_ms:>8.2f} ms")

    await engine.stop()


def export_for_comparison(results_path: Path):
    """导出结果用于与 vLLM/SGLang 对比。"""
    print("\n\n" + "=" * 80)
    print("示例 3: 导出对比格式")
    print("=" * 80)

    # 读取结果
    summary_file = results_path / "benchmark_summary.json"
    if not summary_file.exists():
        print(f"结果文件不存在: {summary_file}")
        return

    # 转换为对比格式
    comparison_data = {
        "framework": "sageLLM",
        "model": "sshleifer/tiny-gpt2",
        "backend": "cpu",
        "workloads": {},
    }

    for workload_name in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]:
        metrics_file = results_path / f"{workload_name}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

            comparison_data["workloads"][workload_name] = {
                "request_throughput_rps": metrics.get("request_throughput_rps", 0),
                "input_throughput_tps": metrics.get("input_throughput_tps", 0),
                "output_throughput_tps": metrics.get("output_throughput_tps", 0),
                "total_throughput_tps": metrics.get("total_throughput_tps", 0),
                "p50_ttft_ms": metrics.get("p50_ttft_ms", 0),
                "p95_ttft_ms": metrics.get("p95_ttft_ms", 0),
                "p99_ttft_ms": metrics.get("p99_ttft_ms", 0),
            }

    # 保存对比格式
    output_file = results_path / "comparison_format.json"
    with open(output_file, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"✓ 对比格式已保存到: {output_file}")
    print("\n对比数据预览:")
    print(json.dumps(comparison_data, indent=2))


async def main():
    """运行所有示例。"""
    print("sageLLM Benchmark - 吞吐量对标示例")
    print("对标参考：vLLM / SGLang")
    print()

    # 示例 1: Batch 模式
    await run_batch_mode_example()

    # 示例 2: Traffic 模式
    await run_traffic_mode_example()

    # 示例 3: 导出对比格式
    export_for_comparison(Path("./outputs/batch_mode_example"))

    print("\n" + "=" * 80)
    print("✓ 所有示例完成！")
    print("=" * 80)
    print("\n如何使用结果：")
    print("1. 查看 ./outputs/batch_mode_example/ 和 ./outputs/traffic_mode_example/")
    print("2. 使用 comparison_format.json 与 vLLM/SGLang 结果对比")
    print("3. 关键指标：request/input/output/total throughput")
    print("\n参考文档：")
    print("- docs/USAGE.md#benchmarking-against-vllmsglang")
    print("- docs/THROUGHPUT_BENCHMARK_PLAN.md")


if __name__ == "__main__":
    asyncio.run(main())
