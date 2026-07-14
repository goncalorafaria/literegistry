import asyncio
from unittest.mock import patch

from fastapi import HTTPException
from literegistry.code_server import (
    CodeRequest,
    CodeResponse,
    StatelessCodeExecutorConfig,
    StatelessCodeExecutorServer,
)


def _server(*, pool_size=1, max_queue_size=1):
    config = StatelessCodeExecutorConfig(
        pool_size=pool_size,
        max_queue_size=max_queue_size,
        preimport_tools=[],
        default_tools=[],
        tool_specs={"math": "math"},
    )
    with patch("literegistry.code_server.get_kvstore", return_value=object()):
        return StatelessCodeExecutorServer(config)


def test_executor_metrics_report_capacity_and_health():
    server = _server(pool_size=2, max_queue_size=1)
    try:
        server.should_run = True
        assert server.check_health() is True

        server._inflight_requests = 3
        metrics = server._executor_metrics()

        assert metrics["active_requests_estimate"] == 2
        assert metrics["queued_requests_estimate"] == 1
        assert metrics["healthy"] is False
    finally:
        server.shutdown()


def test_pool_restart_replaces_only_the_failed_pool():
    server = _server()
    try:
        old_pool = server.process_pool
        asyncio.run(server._restart_process_pool("test failure", old_pool))

        assert server.process_pool is not old_pool
        assert server._pool_restart_count == 1
        assert "test failure" in server._recent_failures[-1]
    finally:
        server.shutdown()


def test_outer_timeout_restarts_pool_and_marks_response_retryable():
    server = _server()

    async def run_test():
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        old_pool = server.process_pool
        with patch.object(loop, "run_in_executor", return_value=pending):
            response = await server.execute_stateless_code(
                "print(1)", max_runtime=0.01
            )

        assert response.success is False
        assert response.retryable is True
        assert server._execution_timeouts == 1
        assert server._pool_restart_count == 1
        assert server.process_pool is not old_pool

    try:
        asyncio.run(run_test())
    finally:
        server.shutdown()


def test_code_response_defaults_to_non_retryable():
    response = CodeResponse(
        stdout="",
        success=False,
        execution_time=0.01,
        stderr="NameError: x",
    )

    assert response.retryable is False


def test_overloaded_executor_rejects_new_request_immediately():
    server = _server(pool_size=1, max_queue_size=0)
    try:
        asyncio.run(server._admission.acquire())
        route = next(route for route in server.app.routes if route.path == "/python")
        try:
            asyncio.run(route.endpoint(CodeRequest(code="print(1)")))
        except HTTPException as exc:
            assert exc.status_code == 503
            assert exc.detail["retryable"] is True
        else:
            raise AssertionError("expected overloaded request to be rejected")
    finally:
        server._admission.release()
        server.shutdown()
