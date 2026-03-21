"""Benchmark runner for sageLLM engines."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sagellm_benchmark.metrics import MetricsAggregator
from sagellm_benchmark.reporters import JSONReporter
from sagellm_benchmark.types import AggregatedMetrics, BenchmarkResult
from sagellm_benchmark.workloads import YEAR1_WORKLOADS, WorkloadConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run.

    Attributes:
        engine: Engine instance to benchmark.
        workloads: List of workload configurations.
        output_dir: Directory for results (default: ./benchmark_results).
        verbose: Enable verbose logging.
        dataset: Optional dataset for sampling real prompts.
        mode: Benchmark mode ('batch' or 'traffic').
    """

    engine: Any  # BaseEngine instance
    workloads: list[WorkloadConfig]
    output_dir: Path = Path("./benchmark_results")
    verbose: bool = False
    dataset: Any = None  # Optional BenchmarkDataset instance
    mode: str = "traffic"  # 'batch' or 'traffic'


class BenchmarkRunner:
    """Runner for executing benchmarks."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self.results: dict[str, AggregatedMetrics] = {}

        # Setup logging
        if config.verbose:
            logging.basicConfig(level=logging.INFO)

    async def run(self) -> dict[str, AggregatedMetrics]:
        """Run all workloads and collect metrics.

        Returns:
            Dictionary mapping workload name to aggregated metrics.
        """
        logger.info(f"Starting benchmark with {len(self.config.workloads)} workloads")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure engine is started
        if not self.config.engine.is_running:
            await self.config.engine.start()

        # Run each workload
        for workload in self.config.workloads:
            logger.info(f"Running workload: {workload.name}")
            aggregated_metrics = await self._run_workload(workload)
            self.results[workload.name] = aggregated_metrics

            # Save individual results using JSONReporter
            output_file = self.config.output_dir / f"{workload.name}_metrics.json"
            reporter = JSONReporter()
            reporter.generate(aggregated_metrics, output_path=str(output_file))
            logger.info(f"Saved metrics to {output_file}")

        # Save summary
        self._save_summary()

        logger.info("Benchmark completed")
        return self.results

    async def _run_workload(self, workload: WorkloadConfig) -> AggregatedMetrics:
        """Run a single workload.

        Args:
            workload: Workload configuration.

        Returns:
            Aggregated metrics from all requests.
        """
        results: list[BenchmarkResult] = []

        # Import Request here to avoid circular dependency
        try:
            from sagellm_protocol import Request
        except ImportError:
            logger.error(
                "sagellm_protocol not installed. Install with: pip install isagellm-protocol"
            )
            raise

        # Get prompts from dataset if available
        if self.config.dataset is not None:
            from sagellm_benchmark.types import WorkloadSpec

            spec = WorkloadSpec(
                name=workload.name,
                workload_type=workload.workload_type,
                prompt_len=workload.prompt_tokens,
                output_len=workload.max_tokens,
                num_requests=workload.num_requests,
            )
            dataset_requests = self.config.dataset.sample(spec)
            prompts = [req.prompt for req in dataset_requests]
        else:
            # Use hardcoded prompt from workload
            prompts = [workload.prompt] * workload.num_requests

        if workload.concurrent:
            # Run requests concurrently
            tasks = []
            for i in range(workload.num_requests):
                request = Request(
                    request_id=f"{workload.name}-{i:03d}",
                    trace_id=f"benchmark-{workload.name}",
                    model=self.config.engine.config.model_path,
                    prompt=prompts[i],
                    max_tokens=workload.max_tokens,
                    temperature=workload.temperature,
                    top_p=workload.top_p,
                    stream=False,
                )
                tasks.append(self.config.engine.execute(request))

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert responses to BenchmarkResult
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Request {i} failed: {response}")
                    # Skip failed requests (could create failed BenchmarkResult in future)
                else:
                    # Convert Response to BenchmarkResult
                    result = self._response_to_result(response)
                    results.append(result)

        else:
            # Run requests sequentially
            for i in range(workload.num_requests):
                request = Request(
                    request_id=f"{workload.name}-{i:03d}",
                    trace_id=f"benchmark-{workload.name}",
                    model=self.config.engine.config.model_path,
                    prompt=prompts[i],
                    max_tokens=workload.max_tokens,
                    temperature=workload.temperature,
                    top_p=workload.top_p,
                    stream=False,
                )

                try:
                    response = await self.config.engine.execute(request)
                    result = self._response_to_result(response)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Request {i} failed: {e}")

        # Aggregate results
        aggregator = MetricsAggregator()
        return aggregator.aggregate(results)

    def _response_to_result(self, response: Any) -> BenchmarkResult:
        """Convert engine Response to BenchmarkResult.

        Args:
            response: Engine response object.

        Returns:
            BenchmarkResult instance.
        """
        # 提取 metrics（Metrics.timestamps 由 llm_engine 直接填充，无需二次注入）
        metrics = response.metrics if hasattr(response, "metrics") else None

        # 提取 output_tokens 数量
        output_tokens_count = 0
        if hasattr(response, "output_tokens") and response.output_tokens:
            raw = response.output_tokens
            output_tokens_count = len(raw) if isinstance(raw, list) else int(raw)

        # 提取 prompt_tokens（llm_engine 目前不填该字段，暂时留 0）
        prompt_tokens = 0
        if hasattr(response, "prompt_tokens") and response.prompt_tokens is not None:
            prompt_tokens = int(response.prompt_tokens)

        # 从 metrics.timestamps 计算 e2e 延迟（ms）
        e2e_latency_ms = 0.0
        if metrics is not None and metrics.timestamps is not None:
            ts = metrics.timestamps
            if ts.queued_at > 0 and ts.completed_at > 0:
                e2e_latency_ms = (ts.completed_at - ts.queued_at) * 1000.0

        return BenchmarkResult(
            request_id=response.request_id,
            success=not hasattr(response, "error") or response.error is None,
            error=response.error if hasattr(response, "error") else None,
            metrics=metrics,
            output_text=response.output_text if hasattr(response, "output_text") else "",
            output_tokens=output_tokens_count,
            prompt_tokens=prompt_tokens,
            e2e_latency_ms=e2e_latency_ms,
        )

    def _save_summary(self) -> None:
        """Save summary of all workloads."""
        summary = {
            "workloads": {},
            "overall": {
                "total_workloads": len(self.results),
                "total_requests": sum(m.total_requests for m in self.results.values()),
                "successful_requests": sum(m.successful_requests for m in self.results.values()),
                "failed_requests": sum(m.failed_requests for m in self.results.values()),
            },
        }

        for name, metrics in self.results.items():
            # Convert AggregatedMetrics to dict for JSON serialization
            summary["workloads"][name] = {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "avg_ttft_ms": metrics.avg_ttft_ms,
                "avg_throughput_tps": metrics.avg_throughput_tps,
                "error_rate": metrics.error_rate,
            }

        import json

        summary_file = self.config.output_dir / "benchmark_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        logger.info(f"Saved summary to {summary_file}")


async def run_year1_benchmark(
    engine: Any, output_dir: Path | str = Path("./benchmark_results")
) -> dict[str, AggregatedMetrics]:
    """Convenience function to run Year 1 Demo Contract workloads.

    Args:
        engine: Engine instance to benchmark.
        output_dir: Output directory for results.

    Returns:
        Dictionary mapping workload name to aggregated metrics.
    """
    config = BenchmarkConfig(
        engine=engine,
        workloads=YEAR1_WORKLOADS,
        output_dir=Path(output_dir),
        verbose=True,
    )

    runner = BenchmarkRunner(config)
    return await runner.run()
