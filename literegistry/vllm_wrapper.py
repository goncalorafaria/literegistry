import fire
import random

from literegistry.executable_wrapper import ExecutableWrapper
from literegistry.runtime import (
    build_runtime,
    default_vllm_env,
    with_default_binds,
    with_default_env,
)


class VLLMServerManager(ExecutableWrapper):
    """VLLM server manager implementation"""

    def get_server_command(self) -> list:
        """Return the command to start vLLM server"""
        if self.runtime.name == "apptainer":
            return ["vllm", "serve"]
        return ["python", "-m", "vllm.entrypoints.openai.api_server"]

    def get_model_args(self) -> list:
        """Return vLLM model arguments."""
        if self.runtime.name == "apptainer":
            return [self.model]
        return super().get_model_args()

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
    runtime: str = "apptainer",
    image: str = "vllm-openai_latest-cu129-ubuntu2404.sif",
    image_source: str = "docker://vllm/vllm-openai:latest-cu129-ubuntu2404",
    pull_image: bool = True,
    workdir: str = None,
    bind=None,
    env=None,
    apptainer_nv: bool = True,
    apptainer_cleanenv: bool = True,
    apptainer_executable: str = "apptainer",
    apptainer_extra_args=None,
    **kwargs,
):
    """
    Run vLLM server with monitoring

    Args:
        model: Model name/path
        host: Server host 
        registry: Directory for server registry
        port: Server port (random if not specified)
        runtime: Launch runtime ("local" or "apptainer")
        image: Apptainer image path when runtime="apptainer"
        image_source: Optional source used by "apptainer pull"
        pull_image: Pull image_source before launch when provided
        bind: Apptainer bind mount(s), e.g. /host:/container
        env: Apptainer environment entry or entries as KEY=VALUE
        **kwargs: Additional arguments to pass to vLLM server (e.g., enable_chunked_prefill=True, max_num_seqs=256)
        
    Example:
        python -m literegistry.vllm --model allenai/Llama-3.1-Tulu-3-8B-DPO --enable_chunked_prefill=True --max_num_seqs=256
    """
    manager = VLLMServerManager(
        model=model,
        port=random.randint(8000, 12000) if port is None else port,
        host=host,
        registry=registry,
        runtime=build_runtime(
            runtime=runtime,
            image=image,
            image_source=image_source,
            pull_image=pull_image,
            workdir=workdir,
            bind=with_default_binds(bind, include_vllm_cache=True),
            env=with_default_env(default_vllm_env(), env),
            apptainer_nv=apptainer_nv,
            apptainer_cleanenv=apptainer_cleanenv,
            apptainer_executable=apptainer_executable,
            apptainer_extra_args=apptainer_extra_args,
        ),
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
