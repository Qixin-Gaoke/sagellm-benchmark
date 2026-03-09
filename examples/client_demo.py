"""Example: Using different benchmark clients.

This example demonstrates how to use various benchmark clients:
1. GatewayClient - For OpenAI-protocol HTTP APIs (sagellm-gateway)
2. VLLMClient - For vLLM backend
3. LMDeployClient - For LMDeploy backend
4. SageLLMClient - For native sagellm-backend engines (no HTTP)

Note: This demo requires running services. For unit tests, see tests/ directory.
"""

from __future__ import annotations

import asyncio
import logging

from sagellm_benchmark.types import BenchmarkRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Note: CPU client demo removed - MockClient was removed in CPU-first refactor
# For testing without real backend, use pytest fixtures in tests/ directory


async def demo_gateway_client() -> None:
    """Demo: GatewayClient for sagellm-gateway."""
    logger.info("=" * 60)
    logger.info("Demo: GatewayClient (requires running sagellm-gateway)")
    logger.info("=" * 60)

    try:
        from sagellm_benchmark.clients.openai_client import GatewayClient
    except ImportError:
        logger.error(
            "GatewayClient dependencies are missing. Install with: pip install -U isagellm-benchmark"
        )
        return

    # Connect to local sagellm-gateway
    client = GatewayClient(
        base_url="http://localhost:8000/v1",
        api_key="benchmark",
        timeout=60.0,
    )

    # Health check
    is_healthy = await client.health_check()
    if not is_healthy:
        logger.warning("Gateway not available, skipping demo")
        await client.close()
        return

    # Create request
    request = BenchmarkRequest(
        prompt="Explain quantum computing in simple terms.",
        max_tokens=200,
        request_id="openai-001",
        model="default",
        temperature=0.8,
    )

    # Execute
    result = await client.generate(request)

    logger.info(f"Success: {result.success}")
    if result.success:
        logger.info(f"Output length: {result.output_tokens} tokens")
        logger.info(f"TTFT: {result.metrics.ttft_ms:.2f}ms" if result.metrics else "No metrics")
    else:
        logger.error(f"Error: {result.error}")

    await client.close()


async def demo_vllm_client() -> None:
    """Demo: VLLMClient for vLLM backend."""
    logger.info("=" * 60)
    logger.info("Demo 3: VLLMClient (server mode)")
    logger.info("=" * 60)

    try:
        from sagellm_benchmark.clients.vllm_client import VLLMClient
    except ImportError:
        logger.error("Required packages not installed")
        return

    # Connect to vLLM server
    client = VLLMClient(
        mode="server",
        base_url="http://localhost:8000/v1",
        timeout=60.0,
    )

    # Health check
    is_healthy = await client.health_check()
    if not is_healthy:
        logger.warning("vLLM server not available, skipping demo")
        await client.close()
        return

    # Create request
    request = BenchmarkRequest(
        prompt="Write a Python function to calculate fibonacci numbers.",
        max_tokens=150,
        request_id="vllm-001",
        model="default",
    )

    # Execute
    result = await client.generate(request)

    logger.info(f"Success: {result.success}")
    if result.success:
        logger.info(f"Output: {result.output_text[:200]}")
        logger.info(
            f"Throughput: {result.metrics.throughput_tps:.2f}tps"
            if result.metrics
            else "No metrics"
        )
    else:
        logger.error(f"Error: {result.error}")

    await client.close()


async def demo_batch_execution() -> None:
    """Demo: Batch execution patterns.

    Note: This demo is disabled - requires a running backend service.
    For batch testing, use GatewayClient with a running sagellm-gateway.
    """
    logger.info("=" * 60)
    logger.info("Demo: Batch execution (DISABLED - requires backend)")
    logger.info("=" * 60)
    logger.info("This demo requires a running backend service.")
    logger.info("Example: Use GatewayClient with sagellm-gateway")
    return

    # NOTE: kept intentionally minimal to avoid stale pseudo-code drift.
    # Batch demonstration should be implemented against real running services only.


async def demo_error_handling() -> None:
    """Demo: Error handling and timeout.

    Note: This demo is disabled - requires a running backend service.
    """
    logger.info("=" * 60)
    logger.info("Demo: Error Handling (DISABLED - requires backend)")
    logger.info("=" * 60)
    logger.info("This demo requires a running backend service.")
    logger.info("Example: Use GatewayClient with error simulation")
    return

    # NOTE: Error-handling demonstration should be implemented against
    # a real running backend service.


async def main() -> None:
    """Run all demos.

    Note: Most demos require running backend services.
    Uncomment the demos below if you have services running.
    """
    logger.info("=" * 60)
    logger.info("Client Demo - Requires Running Services")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Available demos (uncomment to run):")
    logger.info("  - demo_gateway_client(): Requires sagellm-gateway")
    logger.info("  - demo_vllm_client(): Requires vLLM server")
    logger.info("  - demo_lmdeploy_client(): Requires LMDeploy server")
    logger.info("")
    logger.info("For unit testing without services, see tests/ directory")
    logger.info("=" * 60)

    # Uncomment if you have services running:
    # await demo_gateway_client()
    # await demo_vllm_client()
    # await demo_lmdeploy_client()


if __name__ == "__main__":
    asyncio.run(main())
