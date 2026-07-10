import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from literegistry.runtime import (
    ApptainerRuntime,
    ImageSpec,
    build_runtime,
    default_apptainer_image_dir,
    resolve_apptainer_image,
)
import literegistry.sglang_wrapper as sglang_wrapper
from literegistry.sglang_wrapper import SGLangServerManager
import literegistry.vllm_wrapper as vllm_wrapper
from literegistry.vllm_wrapper import VLLMServerManager
import literegistry.redis as redis_wrapper


class RuntimeCommandTest(unittest.TestCase):
    def test_resolve_apptainer_image_uses_home(self):
        with patch.dict("os.environ", {"HOME": "/home/graf"}, clear=True):
            self.assertEqual(
                resolve_apptainer_image("vllm.sif"),
                "/home/graf/vllm.sif",
            )
            self.assertEqual(default_apptainer_image_dir(), "/home/graf")

    def test_resolve_apptainer_image_keeps_absolute_paths(self):
        self.assertEqual(
            resolve_apptainer_image("/shared/images/vllm.sif"),
            "/shared/images/vllm.sif",
        )

    def test_vllm_local_command_matches_existing_entrypoint(self):
        with tempfile.TemporaryDirectory() as registry:
            manager = VLLMServerManager(
                registry=registry,
                model="allenai/model",
                host="0.0.0.0",
                port=8000,
                tensor_parallel_size=1,
                trust_remote_code=True,
                language_model_only=True,
                unused_none=None,
            )

        self.assertEqual(
            manager.build_backend_command(),
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                "allenai/model",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--tensor-parallel-size",
                "1",
                "--trust-remote-code",
                "--language-model-only",
            ],
        )

    def test_vllm_apptainer_command_wraps_vllm_serve(self):
        image = "/gscratch/ark/graf/vllm-openai_latest-cu129-ubuntu2404.sif"
        runtime = ApptainerRuntime(
            ImageSpec(
                image=image,
                source="docker://vllm/vllm-openai:latest-cu129-ubuntu2404",
            ),
            binds="/gscratch/ark/graf:/gscratch/ark/graf",
            env=[
                "HF_HOME=/gscratch/ark/graf/hf_cache",
                "VLLM_CACHE_ROOT=/gscratch/ark/graf/vllm_cache",
            ],
        )

        with tempfile.TemporaryDirectory() as registry:
            manager = VLLMServerManager(
                registry=registry,
                runtime=runtime,
                model="/models/checkpoint",
                host="0.0.0.0",
                port=7248,
                tensor_parallel_size=1,
                dtype="float16",
                max_model_len=4096,
                trust_remote_code=True,
                language_model_only=True,
                safetensors_load_strategy="prefetch",
            )

        self.assertEqual(
            manager.runtime.build_command(manager.build_backend_command()),
            [
                "apptainer",
                "exec",
                "--nv",
                "--cleanenv",
                "--bind",
                "/gscratch/ark/graf:/gscratch/ark/graf",
                "--env",
                "HF_HOME=/gscratch/ark/graf/hf_cache",
                "--env",
                "VLLM_CACHE_ROOT=/gscratch/ark/graf/vllm_cache",
                image,
                "vllm",
                "serve",
                "/models/checkpoint",
                "--host",
                "0.0.0.0",
                "--port",
                "7248",
                "--tensor-parallel-size",
                "1",
                "--dtype",
                "float16",
                "--max-model-len",
                "4096",
                "--trust-remote-code",
                "--language-model-only",
                "--safetensors-load-strategy",
                "prefetch",
            ],
        )

    def test_sglang_apptainer_command_uses_model_path_flag(self):
        with patch.dict("os.environ", {"HOME": "/home/graf"}, clear=True):
            runtime = build_runtime(
                runtime="apptainer",
                image="sglang.sif",
                pull_image=False,
                bind=["/data:/data"],
                env={"HF_HOME": "/cache"},
            )

        with tempfile.TemporaryDirectory() as registry:
            manager = SGLangServerManager(
                registry=registry,
                runtime=runtime,
                model="/models/sglang",
                port=9000,
                tp_size=1,
            )

        self.assertEqual(
            manager.runtime.build_command(manager.build_backend_command()),
            [
                "apptainer",
                "exec",
                "--nv",
                "--cleanenv",
                "--bind",
                "/data:/data",
                "--env",
                "HF_HOME=/cache",
                "/home/graf/sglang.sif",
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                "/models/sglang",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--tp-size",
                "1",
            ],
        )

    def test_redis_apptainer_command_uses_official_image(self):
        with patch.dict("os.environ", {"HOME": "/home/graf"}, clear=True), patch(
            "os.makedirs"
        ), patch(
            "literegistry.runtime.subprocess.run"
        ) as run, patch("literegistry.redis.subprocess.Popen") as popen, patch(
            "literegistry.redis.time.sleep"
        ):
            url = redis_wrapper.start_redis_server(port=6380)

        run.assert_called_once_with(
            [
                "apptainer",
                "pull",
                "/home/graf/redis_7-alpine.sif",
                "docker://redis:7-alpine",
            ],
            check=True,
        )
        popen.assert_called_once_with(
            [
                "apptainer",
                "exec",
                "--cleanenv",
                "/home/graf/redis_7-alpine.sif",
                "redis-server",
                "--save",
                "",
                "--appendonly",
                "no",
                "--port",
                "6380",
                "--protected-mode",
                "no",
            ]
        )
        self.assertTrue(url.startswith("redis://"))
        self.assertTrue(url.endswith(":6380"))

    def test_redis_local_command_uses_host_binary(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "literegistry.redis.shutil.which",
            return_value="/usr/bin/redis-server",
        ), patch("literegistry.redis.subprocess.Popen") as popen, patch(
            "literegistry.redis.time.sleep"
        ), patch("literegistry.redis.socket.getfqdn", return_value="redis-host"), patch(
            "builtins.print"
        ) as print_mock:
            redis_wrapper.start_redis_server(port=6381, runtime="local")

        print_mock.assert_any_call("REDIS_URL=redis://redis-host:6381", flush=True)
        popen.assert_called_once_with(
            [
                "/usr/bin/redis-server",
                "--save",
                "",
                "--appendonly",
                "no",
                "--port",
                "6381",
                "--protected-mode",
                "no",
            ]
        )

    def test_redis_log_appends_url_to_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "redis.log")
            with open(log_path, "w") as log_file:
                log_file.write("redis://old-host:6379\n")

            with patch.dict("os.environ", {}, clear=True), patch(
                "literegistry.redis.shutil.which",
                return_value="/usr/bin/redis-server",
            ), patch("literegistry.redis.subprocess.Popen"), patch(
                "literegistry.redis.time.sleep"
            ), patch("literegistry.redis.socket.getfqdn", return_value="redis-host"):
                redis_wrapper.start_redis_server(
                    port=6381,
                    runtime="local",
                    log=log_path,
                )

            with open(log_path, "r") as log_file:
                self.assertEqual(
                    log_file.readlines(),
                    [
                        "redis://old-host:6379\n",
                        "redis://redis-host:6381\n",
                    ],
                )

    def test_redis_log_creates_missing_parent_dirs_and_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "missing", "nested", "redis.log")

            with patch.dict("os.environ", {}, clear=True), patch(
                "literegistry.redis.shutil.which",
                return_value="/usr/bin/redis-server",
            ), patch("literegistry.redis.subprocess.Popen"), patch(
                "literegistry.redis.time.sleep"
            ), patch("literegistry.redis.socket.getfqdn", return_value="redis-host"):
                redis_wrapper.start_redis_server(
                    port=6383,
                    runtime="local",
                    log=log_path,
                )

            with open(log_path, "r") as log_file:
                self.assertEqual(log_file.read(), "redis://redis-host:6383\n")

    def test_redis_foreground_command_blocks_on_server(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "literegistry.redis.shutil.which",
            return_value="/usr/bin/redis-server",
        ), patch("literegistry.redis.subprocess.run") as run, patch(
            "literegistry.redis.subprocess.Popen"
        ) as popen, patch("literegistry.redis.time.sleep") as sleep, patch(
            "literegistry.redis.socket.getfqdn", return_value="redis-host"
        ), patch("builtins.print") as print_mock:
            redis_wrapper.start_redis_server(
                port=6382,
                runtime="local",
                foreground=True,
            )

        print_mock.assert_any_call("REDIS_URL=redis://redis-host:6382", flush=True)
        print_mock.assert_any_call(
            "Redis server running with URL: redis://redis-host:6382",
            flush=True,
        )
        run.assert_called_once_with(
            [
                "/usr/bin/redis-server",
                "--save",
                "",
                "--appendonly",
                "no",
                "--port",
                "6382",
                "--protected-mode",
                "no",
            ],
            check=True,
        )
        popen.assert_not_called()
        sleep.assert_not_called()

    def test_apptainer_prepare_pulls_when_source_is_set(self):
        with tempfile.TemporaryDirectory() as image_dir:
            with patch.dict(
                "os.environ",
                {"LITEREGISTRY_APPTAINER_IMAGE_DIR": image_dir},
                clear=False,
            ):
                runtime = ApptainerRuntime(
                    ImageSpec(
                        image="image.sif",
                        source="docker://repo/image:tag",
                        pull=True,
                    )
                )

                with patch("literegistry.runtime.subprocess.run") as run:
                    runtime.prepare()

            expected_image = os.path.join(image_dir, "image.sif")
        run.assert_called_once_with(
            [
                "apptainer",
                "pull",
                expected_image,
                "docker://repo/image:tag",
            ],
            check=True,
        )

    def test_apptainer_prepare_skips_pull_when_image_exists(self):
        with patch.dict("os.environ", {"HOME": "/home/graf"}, clear=True):
            runtime = ApptainerRuntime(
                ImageSpec(
                    image="image.sif",
                    source="docker://repo/image:tag",
                    pull=True,
                )
            )

            with patch("os.path.exists", return_value=True), patch(
                "literegistry.runtime.subprocess.run"
            ) as run:
                runtime.prepare()

        run.assert_not_called()

    def test_vllm_apptainer_defaults_to_vllm_openai_image(self):
        shell_env = {
            "HOME": "/home/graf",
            "HF_HOME": "/shared/hf",
            "VLLM_CACHE_ROOT": "/shared/vllm",
        }

        def exists(path):
            return path in {
                "/home/graf",
                "/shared/hf",
                "/shared/hf/hub",
                "/shared/hf/transformers",
                "/shared/vllm",
            }

        with patch.dict("os.environ", shell_env, clear=True), patch(
            "os.path.exists",
            side_effect=exists,
        ), patch.object(
            VLLMServerManager, "run"
        ), patch(
            "literegistry.vllm_wrapper.build_runtime",
            wraps=build_runtime,
        ) as runtime_builder:
            vllm_wrapper.main(runtime="apptainer", port=8000)
            runtime_builder.assert_called_once()
            kwargs = runtime_builder.call_args.kwargs
            runtime = build_runtime(**kwargs)

        self.assertEqual(kwargs["runtime"], "apptainer")
        self.assertEqual(
            kwargs["image"],
            "vllm-openai_latest-cu129-ubuntu2404.sif",
        )
        self.assertEqual(
            runtime.image_spec.image,
            "/home/graf/vllm-openai_latest-cu129-ubuntu2404.sif",
        )
        self.assertEqual(
            kwargs["image_source"],
            "docker://vllm/vllm-openai:latest-cu129-ubuntu2404",
        )
        self.assertEqual(
            kwargs["bind"],
            [
                "/home/graf:/home/graf",
                "/shared/hf:/shared/hf",
                "/shared/hf/hub:/shared/hf/hub",
                "/shared/hf/transformers:/shared/hf/transformers",
                "/shared/vllm:/shared/vllm",
            ],
        )
        self.assertEqual(
            kwargs["env"],
            {
                "HOME": "/home/graf",
                "HF_HOME": "/shared/hf",
                "HUGGINGFACE_HUB_CACHE": "/shared/hf/hub",
                "TRANSFORMERS_CACHE": "/shared/hf/transformers",
                "VLLM_CACHE_ROOT": "/shared/vllm",
            },
        )

    def test_sglang_apptainer_defaults_to_official_image(self):
        shell_env = {
            "HOME": "/home/graf",
            "HF_CACHE": "/shared/hf-cache",
            "HUGGINGFACE_HUB_CACHE": "/shared/hub-cache",
            "TRANSFORMERS_CACHE": "/shared/transformers-cache",
        }

        def exists(path):
            return path in {
                "/home/graf",
                "/shared/hf-cache",
                "/shared/hub-cache",
                "/shared/transformers-cache",
            }

        with patch.dict("os.environ", shell_env, clear=True), patch(
            "os.path.exists",
            side_effect=exists,
        ), patch.object(
            SGLangServerManager, "run"
        ), patch(
            "literegistry.sglang_wrapper.build_runtime",
            wraps=build_runtime,
        ) as runtime_builder:
            sglang_wrapper.main(runtime="apptainer", port=8000)
            runtime_builder.assert_called_once()
            kwargs = runtime_builder.call_args.kwargs
            runtime = build_runtime(**kwargs)

        self.assertEqual(kwargs["runtime"], "apptainer")
        self.assertEqual(kwargs["image"], "sglang_latest.sif")
        self.assertEqual(
            runtime.image_spec.image,
            "/home/graf/sglang_latest.sif",
        )
        self.assertEqual(kwargs["image_source"], "docker://lmsysorg/sglang:latest")
        self.assertEqual(
            kwargs["bind"],
            [
                "/home/graf:/home/graf",
                "/shared/hf-cache:/shared/hf-cache",
                "/shared/hub-cache:/shared/hub-cache",
                "/shared/transformers-cache:/shared/transformers-cache",
            ],
        )
        self.assertEqual(
            kwargs["env"],
            {
                "HOME": "/home/graf",
                "HF_HOME": "/shared/hf-cache",
                "HF_CACHE": "/shared/hf-cache",
                "HUGGINGFACE_HUB_CACHE": "/shared/hub-cache",
                "TRANSFORMERS_CACHE": "/shared/transformers-cache",
            },
        )

    def test_default_cache_env_can_be_extended_and_overridden(self):
        shell_env = {
            "HOME": "/home/graf",
            "HF_HOME": "/shared/hf",
            "HUGGINGFACE_HUB_CACHE": "/shared/hf/hub",
            "TRANSFORMERS_CACHE": "/shared/hf/transformers",
            "VLLM_CACHE_ROOT": "/shared/vllm",
        }

        def exists(path):
            return path in {
                "/home/graf",
                "/shared/hf",
                "/shared/hf/hub",
                "/shared/hf/transformers",
                "/shared/vllm",
                "/data",
            }

        with patch.dict("os.environ", shell_env, clear=True), patch(
            "os.path.exists",
            side_effect=exists,
        ), patch.object(VLLMServerManager, "run"), patch(
            "literegistry.vllm_wrapper.build_runtime"
        ) as runtime_builder:
            runtime_builder.return_value = build_runtime("local")
            vllm_wrapper.main(
                port=8000,
                bind="/data:/data",
                env=[
                    "HF_HOME=/custom/hf",
                    "EXTRA_CACHE=/custom/extra",
                ],
            )

        kwargs = runtime_builder.call_args.kwargs
        self.assertEqual(
            kwargs["bind"],
            [
                "/home/graf:/home/graf",
                "/shared/hf:/shared/hf",
                "/shared/hf/hub:/shared/hf/hub",
                "/shared/hf/transformers:/shared/hf/transformers",
                "/shared/vllm:/shared/vllm",
                "/data:/data",
            ],
        )
        self.assertEqual(kwargs["env"]["HF_HOME"], "/custom/hf")
        self.assertEqual(kwargs["env"]["EXTRA_CACHE"], "/custom/extra")
        self.assertEqual(
            kwargs["env"]["HUGGINGFACE_HUB_CACHE"],
            "/shared/hf/hub",
        )

    def test_default_binds_skip_missing_cache_subdirectories(self):
        shell_env = {
            "HOME": "/home/graf",
            "HF_HOME": "/shared/hf",
            "VLLM_CACHE_ROOT": "/shared/vllm",
        }

        def exists(path):
            return path in {
                "/home/graf",
                "/shared/hf",
                "/shared/vllm",
            }

        with patch.dict("os.environ", shell_env, clear=True), patch(
            "os.path.exists",
            side_effect=exists,
        ), patch.object(VLLMServerManager, "run"), patch(
            "literegistry.vllm_wrapper.build_runtime"
        ) as runtime_builder:
            runtime_builder.return_value = build_runtime("local")
            vllm_wrapper.main(port=8000)

        self.assertEqual(
            runtime_builder.call_args.kwargs["bind"],
            [
                "/home/graf:/home/graf",
                "/shared/hf:/shared/hf",
                "/shared/vllm:/shared/vllm",
            ],
        )


if __name__ == "__main__":
    unittest.main()
