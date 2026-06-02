import fire
import random

from literegistry.executable_wrapper import ExecutableWrapper
from literegistry.runtime import (
    build_runtime,
    default_hf_env,
    with_default_binds,
    with_default_env,
)
import os
# python3 -m sglang.launch_server 
class SGLangServerManager(ExecutableWrapper):
    """SGLang server manager implementation"""

    def get_server_command(self) -> list:
        """Return the command to start SGLang server"""
        return ["python3", "-m", "sglang.launch_server"]

    def get_model_flag(self) -> str:
        """Return the model flag for SGLang"""
        return "--model-path"

    def get_server_name(self) -> str:
        """Return the server name"""
        return "SGLang"


def main(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    host: str = "0.0.0.0", 
    registry: str = "/mmfs1/home/gfaria/registry",
    port: int = None,
    runtime: str = "apptainer",
    image: str = "sglang_latest.sif",
    image_source: str = "docker://lmsysorg/sglang:latest",
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
    Run SGLang server with monitoring

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
        **kwargs: Additional arguments to pass to SGLang server (e.g., tp_size=1, mem_fraction_static=0.8)
        
    Example:
        python -m literegistry.sglang --model allenai/Llama-3.1-Tulu-3-8B-DPO --tp_size=1
    """
    
    ## need to unset all proxies 
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("no_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    
    """
    unset http_proxy
    unset https_proxy
    unset HTTP_PROXY
    unset HTTPS_PROXY
    """

    manager = SGLangServerManager(
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
            bind=with_default_binds(bind),
            env=with_default_env(default_hf_env(), env),
            apptainer_nv=apptainer_nv,
            apptainer_cleanenv=apptainer_cleanenv,
            apptainer_executable=apptainer_executable,
            apptainer_extra_args=apptainer_extra_args,
        ),
        **kwargs,
    )
    manager.run()


if __name__ == "__main__":
    """python -m sglang.launch_server --model-path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size 1 \
    --mem-fraction-static 0.9
    """

    fire.Fire(main)
