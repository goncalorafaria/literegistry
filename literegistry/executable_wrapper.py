import subprocess
import requests
import time
import socket
import threading
import asyncio
from abc import ABC, abstractmethod

from literegistry import ServerRegistry, get_kvstore


class ExecutableWrapper(ABC):
    """Abstract base class for LLM server managers (vLLM, SGLang, etc.)"""

    def __init__(
        self,
        registry: str,
        model: str = "allenai/Llama-3.1-Tulu-3-8B-DPO",
        port: int = 8000,
        host: str = "0.0.0.0",
        max_history=3600,
        heartbeat_interval=30,
        **kwargs,
    ):
        """
        Initialize server manager

        Args:
            model: Model name/path
            port: Server port
            host: Server host
            registry: Directory for server registry 
            max_history: Maximum history for registry
            **kwargs: Additional arguments to pass to the server
        """
        self.model = model
        self.port = port
        self.host = host 
        self.metrics_port = self.port
        self.url = f"http://{socket.getfqdn()}"
        self.extra_kwargs = kwargs
        self.heartbeat_interval = heartbeat_interval

        # Initialize registry
        store = get_kvstore(registry)
        self.registry = ServerRegistry(
            store=store,
            max_history=max_history,
        )

        self.process = None
        self.should_run = True

    @abstractmethod
    def get_server_command(self) -> list:
        """
        Return the base command to start the server.
        
        Returns:
            list: Command components like ["python", "-m", "module.name"]
        """
        pass

    @abstractmethod
    def get_model_flag(self) -> str:
        """
        Return the command-line flag for specifying the model.
        
        Returns:
            str: Flag name like "--model" or "--model-path"
        """
        pass

    @abstractmethod
    def get_server_name(self) -> str:
        """
        Return the name of the server for logging.
        
        Returns:
            str: Server name like "vLLM" or "SGLang"
        """
        pass

    def start_server(self):
        """Start the server as a subprocess"""
        cmd = self.get_server_command()
        cmd.extend([
            self.get_model_flag(),
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ])

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
        #log_filename = f"{self.get_server_name().lower()}_server_{self.registry.server_id}.log"
        #print(log_filename)
        #   log_file = open(log_filename, "w")
        self.process = subprocess.Popen(
            cmd, stdout=None, stderr=None, universal_newlines=True
        )
        print(f"Started {self.get_server_name()} server with PID {self.process.pid}")

        # Register server with metadata
        metadata = {
            "model_path": self.model,
            "host": self.host,
            "port": self.port,
            "backend": self.get_server_name().lower(),  # "vllm" or "sglang"
            "extra_kwargs": self.extra_kwargs,
        }
        
        asyncio.run(
            self.registry.register_server(
                url=self.url,
                port=self.port, 
                metadata=metadata
            )
        )

    def check_health(self):
        """Check if server is responding"""
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
            time.sleep(self.heartbeat_interval)

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

