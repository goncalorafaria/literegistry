import asyncio
from unittest.mock import patch

import pytest

from literegistry.terminal_server import (
    PipelineLimitError,
    PipelineValidationError,
    TerminalPipelineServer,
    TerminalRequest,
    TerminalServerConfig,
    parse_pipeline,
)


def _server():
    with patch("literegistry.terminal_server.get_kvstore", return_value=object()):
        return TerminalPipelineServer(TerminalServerConfig())


def test_parse_pipeline_accepts_stdin_only_log_slicing():
    assert parse_pipeline("rg -i error | head -n 10") == [
        ["rg", "-i", "error"],
        ["head", "-n", "10"],
    ]


def test_parse_pipeline_accepts_grep_context_option():
    assert parse_pipeline('grep -n "Examples of Successful" -C 2') == [
        ["grep", "-n", "Examples of Successful", "-C", "2"]
    ]


def test_parse_pipeline_accepts_attached_grep_context_options():
    assert parse_pipeline('grep -i -A3 -B3 "Examples of Successful"') == [
        ["grep", "-i", "-A3", "-B3", "Examples of Successful"]
    ]


def test_parse_pipeline_accepts_head_byte_count():
    assert parse_pipeline("head -c 2000") == [["head", "-c", "2000"]]


def test_parse_pipeline_accepts_wc_line_count():
    assert parse_pipeline("wc -l") == [["wc", "-l"]]


def test_parse_pipeline_accepts_echo():
    assert parse_pipeline('echo -n "hello"') == [["echo", "-n", "hello"]]
    assert parse_pipeline("echo hi | cat") == [["echo", "hi"], ["cat"]]


def test_parse_pipeline_rejects_echo_paths():
    with pytest.raises(PipelineValidationError):
        parse_pipeline("echo /etc/passwd")


def test_parse_pipeline_accepts_stdin_only_cat_and_awk_line_limit():
    assert parse_pipeline("cat | head -n 200") == [["cat"], ["head", "-n", "200"]]
    assert parse_pipeline("awk 'NR<=200'") == [["awk", "NR<=200"]]
    assert parse_pipeline("awk '{print substr($0,1,500); exit}'") == [
        ["awk", "{print substr($0,1,500); exit}"]
    ]
    assert parse_pipeline("nl -ba | tail -n +1") == [
        ["nl", "-ba"],
        ["tail", "-n", "+1"],
    ]


def test_parse_pipeline_accepts_combined_grep_flags_and_stdin_sentinel():
    assert parse_pipeline('grep -iE "January|February"') == [
        ["grep", "-i", "-E", "January|February"]
    ]
    assert parse_pipeline("rg . - | head -n 200") == [
        ["rg", ".", "-"],
        ["head", "-n", "200"],
    ]


def test_parse_pipeline_accepts_trailing_or_true_as_a_noop():
    assert parse_pipeline("rg -i turnout || true") == [["rg", "-i", "turnout"]]


def test_parse_pipeline_preserves_quoted_regex_alternation():
    assert parse_pipeline('grep -i -E "queries|keys|values|tiling|kernel" -n') == [
        ["grep", "-i", "-E", "queries|keys|values|tiling|kernel", "-n"]
    ]


def test_parse_pipeline_preserves_quoted_literal_pipe():
    assert parse_pipeline('grep -n "|"') == [["grep", "-n", "|"]]


def test_parse_pipeline_preserves_quoted_angle_brackets():
    assert parse_pipeline('grep -n "<cite" -m 10') == [
        ["grep", "-n", "<cite", "-m", "10"]
    ]


def test_parse_pipeline_accepts_safe_sed_address_with_ordinary_e_text():
    assert parse_pipeline(r"sed -n '/## Sartorial\/style changes/,$p'") == [
        ["sed", "-n", r"/## Sartorial\/style changes/,$p"]
    ]


@pytest.mark.parametrize(
    "pipeline",
    [
        "rg error; cat /etc/passwd",
        "grep error < /etc/passwd",
        "awk 'system(\"id\")'",
        "sed '1e id'",
        "jq -f /etc/passwd",
        "rg error /var/log/messages",
    ],
)
def test_parse_pipeline_rejects_shell_and_file_access(pipeline):
    with pytest.raises(PipelineValidationError):
        parse_pipeline(pipeline)


def test_terminal_server_executes_a_safe_pipeline():
    server = _server()
    response = asyncio.run(
        server.execute(
            TerminalRequest(
                contents="first\nsecond\nthird\n",
                command="head -n 2 | tail -n 1",
            )
        )
    )

    assert response.success is True
    assert response.stdout == "second\n"
    assert response.exit_code == 0


def test_terminal_server_enforces_output_limit():
    with patch("literegistry.terminal_server.get_kvstore", return_value=object()):
        server = TerminalPipelineServer(
            TerminalServerConfig(max_output_bytes=4, max_stderr_bytes=1024)
        )

    with pytest.raises(PipelineLimitError):
        asyncio.run(
            server.execute(
                TerminalRequest(contents="first\nsecond\n", command="head -n 2")
            )
        )


def test_terminal_metadata_registers_terminal_model():
    server = _server()

    metadata = server._metadata()

    assert metadata["model_path"] == "terminal"
    assert "rg" in metadata["extra_kwargs"]["commands"]
    assert "echo" in metadata["extra_kwargs"]["commands"]


def test_response_truncation_appends_missing_character_marker():
    server = _server()

    stdout, truncated, missing = server._truncate_stdout("abcdef", 3)

    assert stdout == "abc\n[ truncated (3 characters missing) ]"
    assert truncated is True
    assert missing == 3


def test_response_truncation_cannot_exceed_server_limit():
    with patch("literegistry.terminal_server.get_kvstore", return_value=object()):
        server = TerminalPipelineServer(TerminalServerConfig(max_response_chars=3))

    with pytest.raises(PipelineValidationError, match="server maximum"):
        server._truncate_stdout("abcdef", 4)
