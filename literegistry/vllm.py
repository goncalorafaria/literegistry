import subprocess
import requests
import time
import socket
import fire
import threading
import asyncio
from typing import Tuple
import re
import math
import random


from literegistry import ServerRegistry, get_kvstore


def parse_vllm_requests(metrics_text: str) -> Tuple[float, float]:
    """
    Parse VLLM metrics text and extract running and waiting requests.

    Args:
        metrics_text (str): Raw prometheus-style metrics text

    Returns:
        Tuple[float, float]: A tuple containing (running_requests, waiting_requests)

    Example:
        running, waiting = parse_vllm_requests(metrics_text)
        total_requests = running + waiting
    """
    # Pattern to match metric lines with their values
    pattern = r"vllm:num_requests_(\w+){.*?} (-?\d+\.?\d*)"

    # Find all matches in the text
    matches = re.finditer(pattern, metrics_text)

    # Initialize values
    metrics = {"running": 0.0, "waiting": 0.0}

    # Extract values from matches
    for match in matches:
        metric_type = match.group(1)
        if metric_type in ["running", "waiting"]:
            # print(match.group(2))
            metrics[metric_type] = math.floor(float(match.group(2)))

    return metrics["running"] + metrics["waiting"]


class VLLMServerManager:

    def __init__(
        self,
        registry: str,
        model: str = "allenai/Llama-3.1-Tulu-3-8B-DPO",
        port: int = 8000,
        host: str = "0.0.0.0",
      
 
        max_history=3600,
        **kwargs,
    ):
        """
        Initialize VLLM server manager

        Args:
            model: Model name/path
            port: Server port
            host: Server host
            registry: Directory for server registry 
            max_history: Maximum history for registry
            **kwargs: Additional arguments to pass to vLLM server (e.g., enable_chunked_prefill=True)
        """
        self.model = model

        self.port = port
        self.host = host 
        self.metrics_port = self.port
       
        self.url = f"http://{socket.getfqdn()}"
        self.extra_kwargs = kwargs  # Store extra kwargs for vLLM server

        # Initialize registry
        
        store=get_kvstore(registry)
            
        self.registry = ServerRegistry(
            store=store,
            max_history=max_history,
        )

        self.process = None
        self.should_run = True

    def start_server(self):
        """Start the vLLM server as a subprocess"""
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",  #    entrypoints/api_server.py # openai.
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # cmd.extend(["--prometheus-port", str(self.metrics_port)])  # or any other port

        # Add extra kwargs to the command
        for key, value in self.extra_kwargs.items():
            # Convert underscore to hyphen for command-line arguments
            arg_name = f"--{key.replace('_', '-')}"
            
            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    cmd.append(arg_name)
            # Handle None values (skip them)
            elif value is not None:
                cmd.extend([arg_name, str(value)])

        print(cmd)
        print(f"vllm_server_{self.registry.server_id}.log")
        log_file = open(f"vllm_server_{self.registry.server_id}.log", "w")
        self.process = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, universal_newlines=True
        )
        print(f"Started vLLM server with PID {self.process.pid}")

        # Register server with metadata
        metadata = {
            "model_path": self.model,
            "host": self.host,
            "port": self.port,
            "extra_kwargs": self.extra_kwargs,
        }
        
        """
        "route": "v1/completions",
            "args": [
                "prompt",
                "model",
                "max_tokens",
                "temperature",
                "stop",
                "logprobs",
            ],
        """
        
        asyncio.run(
            self.registry.register_server(
                url=self.url,
                port=self.port, 
                metadata=metadata
            )
        )

    def check_health(self):
        """Check if vLLM server is responding"""
        try:
            response = requests.get(f"http://localhost:{self.port}/v1/models")
            return response.status_code == 200
        except requests.exceptions.RequestException:

            return False

    def heartbeat_loop(self):
        """Run heartbeat in a loop"""
        while self.should_run:
            if self.check_health():
                asyncio.run(self.registry.heartbeat(self.url, self.port))
                # print("Heartbeat sent. Status: healthy")
            else:
                print("Server unhealthy!")
            time.sleep(10)

    def cleanup(self):
        """Clean up resources"""
        self.should_run = False
        asyncio.run(self.registry.deregister())
        if self.process:
            self.process.terminate()
            self.process.wait()
        print("Server stopped and deregistered")

    def run(self):
        """Run server and monitoring"""
        try:
            self.start_server()
            print("Waiting for server to initialize...")
            time.sleep(30)  # Wait for model to load

            # Start heartbeat in background thread
            heartbeat_thread = threading.Thread(target=self.heartbeat_loop)
            heartbeat_thread.daemon = True
            heartbeat_thread.start()

            # Wait for shutdown signal
            self.process.wait()

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

def main(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",  # "allenai/Llama-3.1-Tulu-3-8B-DPO",  # allenai/Llama-3.1-Tulu-3-8B-SFT
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
        python vllm_wrapper.py --model allenai/Llama-3.1-Tulu-3-8B-DPO --enable_chunked_prefill=True --max_num_seqs=256
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