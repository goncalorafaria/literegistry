import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union


EnvInput = Optional[Union[Dict[str, str], str, Sequence[str]]]
ListInput = Optional[Union[str, Sequence[str]]]

@dataclass
class ImageSpec:
    """Container image configuration."""

    image: str
    source: Optional[str] = None
    pull: bool = True
    workdir: Optional[str] = None


class Runtime(ABC):
    """Launch runtime for model server commands."""

    name = "runtime"

    def prepare(self) -> None:
        """Prepare the runtime before launching the server."""

    @abstractmethod
    def build_command(self, backend_command: Sequence[str]) -> List[str]:
        """Return the final subprocess command."""

    def metadata(self) -> Dict[str, object]:
        """Return runtime metadata for the registry."""
        return {"runtime": self.name}


class LocalRuntime(Runtime):
    """Run backend commands directly in the current environment."""

    name = "local"

    def build_command(self, backend_command: Sequence[str]) -> List[str]:
        return list(backend_command)


class ContainerRuntime(Runtime):
    """Base class for runtimes that execute commands inside an image."""

    def __init__(self, image_spec: ImageSpec):
        self.image_spec = image_spec

    def metadata(self) -> Dict[str, object]:
        metadata = super().metadata()
        metadata.update(
            {
                "image": self.image_spec.image,
                "image_source": self.image_spec.source,
            }
        )
        return metadata


class ApptainerRuntime(ContainerRuntime):
    """Run backend commands through Apptainer."""

    name = "apptainer"

    def __init__(
        self,
        image_spec: ImageSpec,
        binds: ListInput = None,
        env: EnvInput = None,
        nv: bool = True,
        cleanenv: bool = True,
        executable: str = "apptainer",
        extra_args: ListInput = None,
    ):
        super().__init__(image_spec)
        self.binds = _normalize_list(binds)
        self.env = _normalize_env(env)
        self.nv = nv
        self.cleanenv = cleanenv
        self.executable = executable
        self.extra_args = _normalize_list(extra_args)

    def prepare(self) -> None:
        if self.image_spec.pull and self.image_spec.source:
            if os.path.exists(self.image_spec.image):
                print(f"Using existing Apptainer image: {self.image_spec.image}")
                return
            subprocess.run(
                [
                    self.executable,
                    "pull",
                    self.image_spec.image,
                    self.image_spec.source,
                ],
                check=True,
            )

    def build_command(self, backend_command: Sequence[str]) -> List[str]:
        command = [self.executable, "exec"]
        if self.nv:
            command.append("--nv")
        if self.cleanenv:
            command.append("--cleanenv")

        for bind in self.binds:
            command.extend(["--bind", bind])

        for key, value in self.env.items():
            command.extend(["--env", f"{key}={value}"])

        if self.image_spec.workdir:
            command.extend(["--pwd", self.image_spec.workdir])

        command.extend(self.extra_args)
        command.append(self.image_spec.image)
        command.extend(backend_command)
        return command


def build_runtime(
    runtime: Union[str, Runtime] = "local",
    image: Optional[str] = None,
    image_source: Optional[str] = None,
    pull_image: bool = True,
    workdir: Optional[str] = None,
    bind: ListInput = None,
    env: EnvInput = None,
    apptainer_nv: bool = True,
    apptainer_cleanenv: bool = True,
    apptainer_executable: str = "apptainer",
    apptainer_extra_args: ListInput = None,
) -> Runtime:
    """Construct a runtime from CLI-friendly values."""
    if isinstance(runtime, Runtime):
        return runtime

    runtime_name = runtime.lower()
    if runtime_name == "local":
        return LocalRuntime()
    if runtime_name == "apptainer":
        if image is None:
            raise ValueError("--image is required when --runtime=apptainer")
        return ApptainerRuntime(
            image_spec=ImageSpec(
                image=image,
                source=image_source,
                pull=pull_image,
                workdir=workdir,
            ),
            binds=bind,
            env=env,
            nv=apptainer_nv,
            cleanenv=apptainer_cleanenv,
            executable=apptainer_executable,
            extra_args=apptainer_extra_args,
        )

    raise ValueError(f"Unsupported runtime: {runtime}")


def _normalize_list(value: ListInput) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def default_hf_env() -> Dict[str, str]:
    """Return Hugging Face cache env derived from the launching shell."""
    home = os.environ.get("HOME") or os.path.expanduser("~")
    hf_home = (
        os.environ.get("HF_HOME")
        or os.environ.get("HF_CACHE")
        or os.path.join(home, ".cache", "huggingface")
    )
    hub_cache = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or os.path.join(hf_home, "hub")
    )
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE") or os.path.join(
        hf_home,
        "transformers",
    )

    env = {
        "HOME": home,
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": hub_cache,
        "TRANSFORMERS_CACHE": transformers_cache,
    }
    if os.environ.get("HF_CACHE"):
        env["HF_CACHE"] = os.environ["HF_CACHE"]
    if os.environ.get("HF_HUB_CACHE"):
        env["HF_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    return env


def default_vllm_env() -> Dict[str, str]:
    """Return vLLM cache env derived from the launching shell."""
    env = default_hf_env()
    env["VLLM_CACHE_ROOT"] = os.environ.get(
        "VLLM_CACHE_ROOT",
        os.path.join(env["HOME"], ".cache", "vllm"),
    )
    return env


def default_binds(include_vllm_cache: bool = False) -> List[str]:
    """Bind existing shell-derived home and cache directories into the container."""
    env = default_vllm_env() if include_vllm_cache else default_hf_env()
    paths = [
        env.get("HOME"),
        env.get("HF_CACHE"),
        env.get("HF_HOME"),
        env.get("HUGGINGFACE_HUB_CACHE"),
        env.get("HF_HUB_CACHE"),
        env.get("TRANSFORMERS_CACHE"),
    ]
    if include_vllm_cache:
        paths.append(env.get("VLLM_CACHE_ROOT"))

    binds = []
    for path in paths:
        if path and os.path.isabs(path) and os.path.exists(path):
            bind = f"{path}:{path}"
            if bind not in binds:
                binds.append(bind)
    return binds


def with_default_binds(value: ListInput, include_vllm_cache: bool = False) -> List[str]:
    """Return shell-derived bind defaults plus any user-provided binds."""
    binds = default_binds(include_vllm_cache=include_vllm_cache)
    for bind in _normalize_list(value):
        if bind not in binds:
            binds.append(bind)
    return binds


def with_default_env(defaults: Dict[str, str], value: EnvInput) -> Dict[str, str]:
    """Return default container env merged with user overrides."""
    env = dict(defaults)
    env.update(_normalize_env(value))
    return env


def _normalize_env(value: EnvInput) -> Dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): str(env_value) for key, env_value in value.items()}

    entries = _normalize_list(value)
    env = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Environment entries must be KEY=VALUE: {entry}")
        key, env_value = entry.split("=", 1)
        env[key] = env_value
    return env
