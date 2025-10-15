import fire
import random
import re
import math
from typing import Tuple

from literegistry.executable_wrapper import ExecutableWrapper


class VLLMServerManager(ExecutableWrapper):
    """VLLM server manager implementation"""

    def get_server_command(self) -> list:
        """Return the command to start vLLM server"""
        return ["python", "-m", "vllm.entrypoints.openai.api_server"]

    def get_model_flag(self) -> str:
        """Return the model flag for vLLM"""
        return "--model"

    def get_server_name(self) -> str:
        """Return the server name"""
        return "vLLM"


def main(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    host: str = "0.0.0.0", 
    registry: str = "/gscratch/ark/graf/registry",
    port: int = None,
    **kwargs,
):
    """
    Run vLLM server with monitoring

    Args:
        model: Model name/path
        host: Server host 
        registry: Directory for server registry
        port: Server port (random if not specified)
        **kwargs: Additional arguments to pass to vLLM server (e.g., enable_chunked_prefill=True, max_num_seqs=256)
        
    Example:
        python -m literegistry.vllm --model allenai/Llama-3.1-Tulu-3-8B-DPO --enable_chunked_prefill=True --max_num_seqs=256
    """
    manager = VLLMServerManager(
        model=model,
        port=random.randint(8000, 12000) if port is None else port,
        host=host,
        registry=registry,
        **kwargs,
    )
    manager.run()


if __name__ == "__main__":
    """python -m vllm.entrypoints.openai.api_server  --model allenai/Llama-3.1-Tulu-3-8B-DPO \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95
    """

    fire.Fire(main)
