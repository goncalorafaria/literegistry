# Introducing LiteRegistry: Lightweight Service Discovery for Distributed Model Inference

> **TL;DR**: LiteRegistry is a lightweight service discovery system for distributed ML inference. It's like Kubernetes service discovery, but without the containers, YAML files, or complexity. Built for HPC research clusters where nodes come and go (checkpoint resources, Slurm preemption, spot instances). Just add `--registry redis://your-host:6379` to your vLLM/SGLang commands and get automatic failover, load balancing, and dynamic scaling. `pip install literegistry` and you're done.

---

## The Reality of Research Computing Today

If you're a researcher or student in a modern ML lab, this scenario will sound familiar:

Your lab has a Slurm cluster. You get some dedicated GPU resources—maybe a few nodes that are "yours" most of the time. But for the big jobs? You're competing for a shared pool of checkpoint resources. These GPUs can be allocated to you when they're free, but at any moment, they can be taken away for higher-priority jobs. Your training run that was humming along on 32 GPUs suddenly drops to 8. Or worse, that inference server you stood up on a borrowed node just disappears.

This is the reality of academic and industry research computing: **your infrastructure is constantly in flux**. Jobs pop up and drop at any moment. Resources come and go. The cluster you have at 2 PM looks nothing like the cluster you have at 2 AM.

Traditional distributed computing tools weren't built for this world. They assume a stable cluster where you request N nodes, get N nodes, and keep those N nodes until your job finishes. They treat node failures as exceptional cases, not everyday occurrences. They require manual intervention to handle dynamic resource allocation.

### What About Kubernetes and Other Orchestration Tools?

You might be thinking: "Doesn't Kubernetes solve this?" And you'd be right—for cloud environments. Kubernetes, Docker Swarm, and similar tools have been the standard for service discovery and orchestration in cloud computing for years. They're battle-tested, feature-rich, and widely adopted.

But here's the problem: **they were designed for containerized cloud environments, not HPC research clusters.**

These tools bring enormous complexity:
- Container orchestration when you just need to run Python scripts
- YAML configuration files that are a full-time job to maintain
- Docker/container knowledge that not every researcher has
- Complete infrastructure overhaul to adopt
- Complexity designed for microservices, not ML workloads

When you're a grad student who just wants to run vLLM on some GPUs and have it work when nodes come and go, containerizing your entire workflow is overkill. You don't want to learn Kubernetes, write Dockerfiles, set up a container registry, and completely change how you deploy your research code.

**What if you could get the core benefits—service discovery, health checking, automatic failover—without changing how you work?**

**This is why LiteRegistry exists.**

LiteRegistry gives you the orchestration capabilities you need, written in Python, designed for ML workloads, and simple enough to add to your existing Slurm scripts without rewriting everything. No containers required. No YAML files. No infrastructure overhaul. Just `pip install literegistry` and get back to your research.

## The Problem: Coordinating Constantly-Changing Distributed Infrastructure

Running large language models at scale in distributed environments presents unique challenges—especially when your infrastructure is dynamic. When you have multiple GPU nodes running model servers across an HPC cluster that's constantly changing, how do you:

- Keep track of which servers are currently available (not which *were* available 5 minutes ago)?
- Route client requests to the right server when nodes can disappear mid-request?
- Handle failures gracefully when "failure" is just Tuesday afternoon?
- Dynamically scale up when checkpoint resources become available?
- Scale down gracefully when resources are reclaimed?

Imagine you're running a research cluster with dozens of GPU nodes, each hosting different models or multiple replicas of the same model. Your clients need to:

- **Discover** which models are currently available
- **Route** requests to the appropriate server
- **Load balance** across multiple replicas
- **Handle failures** when a server goes down
- **Monitor** performance across the cluster

Without a coordination layer, this quickly becomes a nightmare. You'd need to hardcode endpoints, manually update configurations when servers go down, build your own retry logic, and somehow handle the case where half your nodes just got preempted.

<diagram of distributed cluster showing multiple GPU nodes, each running vLLM/SGLang servers, all disconnected with question marks between them - illustrating the chaos without a registry. Show some nodes fading out/disappearing to represent preemption>

**LiteRegistry provides the missing coordination layer** for this dynamic, ever-changing world.

## What is LiteRegistry?

LiteRegistry is a lightweight service registry and discovery system built specifically for distributed model inference deployments. It's designed from the ground up to handle the reality of modern research computing: resources that come and go, jobs that scale up and down, and failures that are part of normal operation.

Think of it as a phonebook for your distributed model servers—one that automatically updates itself when servers appear or disappear, routes traffic intelligently, and handles the chaos of a shared HPC environment so you don't have to.

With first-class support for popular inference engines like **vLLM** and **SGLang**, LiteRegistry makes it simple to:

- **Deploy** model servers that automatically register themselves
- **Route** requests to healthy, available servers
- **Scale** your deployment up or down as resources change
- **Handle failures** without manual intervention
- **Monitor** your entire distributed system from a single dashboard

---

**💡 Key Insight for Research Labs:**

If your lab has a mix of dedicated nodes and checkpoint/preemptible nodes (like most academic clusters do), LiteRegistry is built for you. Submit separate Slurm jobs for your dedicated partition and your checkpoint partition. When checkpoint nodes get preempted, LiteRegistry automatically removes them from the pool. When they come back, resubmit the job and they rejoin automatically. Your clients never need to know what's happening behind the scenes.

---

## How LiteRegistry Works

In a typical research lab scenario, you might have:
- 4 dedicated GPU nodes running your primary inference servers
- 8-16 checkpoint nodes that come and go throughout the day
- Multiple students and researchers sending inference requests
- Jobs being scheduled and preempted by Slurm constantly

LiteRegistry handles all of this dynamically. Here's how:

LiteRegistry consists of four main components that work together:

### 1. The Registry (Key-Value Store)

At the heart of LiteRegistry is a distributed key-value store that tracks all your model servers. Think of it as a phonebook for your cluster—it knows which models are running, where they're located, their health status, and performance metrics.

<diagram of central registry (Redis or FileSystem) with metadata storage showing service entries. Show entries dynamically appearing/fading to represent nodes joining and leaving. Fields shown: model_name, endpoint, health_status, last_heartbeat, latency_metrics>

The registry supports two backends:

**FileSystem Backend**: Perfect for single-node setups or HPC clusters with shared filesystems (NFS). Simple to set up with zero dependencies, but can bottleneck under high concurrency when you have many services registering and querying simultaneously.

**Redis Backend**: Recommended for production deployments, especially when running across multiple nodes without shared storage. Provides high-performance concurrent access and is built for distributed systems.

### 2. Model Server Wrappers (vLLM & SGLang)

LiteRegistry provides wrappers for popular inference engines that handle all the registration complexity for you. When you launch a vLLM or SGLang server through LiteRegistry, it automatically:

- **Registers** itself with the registry on startup
- **Sends heartbeats** to maintain its active status
- **Reports metrics** like request counts and latency
- **Deregisters** gracefully on shutdown

<diagram of vLLM/SGLang server lifecycle: startup -> register with registry -> periodic heartbeat -> serve requests -> report metrics -> shutdown/deregister>

This means you can spin up and tear down servers dynamically, and the system adapts automatically.

### 3. Gateway Server

The Gateway is an HTTP reverse proxy that sits between your clients and model servers. It provides:

- **OpenAI-compatible API** endpoints (`/v1/completions`, `/v1/chat/completions`, `/v1/models`)
- **Automatic load balancing** based on server latency
- **Smart routing** based on the model parameter in requests
- **Health monitoring** and failover

<diagram of request flow: Client -> Gateway Server -> (queries registry for available servers) -> selects best server based on latency -> routes to vLLM/SGLang instance -> returns response>

The Gateway continuously tracks which servers are healthy and routes requests to the fastest available instance. If a server fails, it automatically retries on another replica.

### 4. Client Library & CLI

For programmatic access, LiteRegistry provides:

- **RegistryClient**: Register servers and query available models
- **RegistryHTTPClient**: Make requests with automatic failover and retry logic

For monitoring, there's a CLI tool:

```bash
literegistry summary --registry redis://your-host:6379
```

<diagram of monitoring dashboard showing: multiple models, server counts per model, health status, request throughput, latency percentiles>

## Getting Started: Installation

Installation is straightforward via pip:

```bash
pip install literegistry
```

## Running LiteRegistry: Complete Workflow

Let's walk through deploying a distributed inference cluster on an HPC system.

### Step 1: Start the Registry

First, you need a central registry. For production deployments, use Redis:

```bash
# Start Redis server (or use an existing Redis instance)
literegistry redis --port 6379
```

The Redis server will run on your login node or a dedicated service node. All other components will connect to this registry.

For development or shared filesystem environments, you can skip this step and use a filesystem path instead (e.g., `/shared/registry`).

### Step 2: Launch Model Servers

Now spin up your vLLM or SGLang servers. The beauty of LiteRegistry is that you can use **all standard vLLM/SGLang arguments**—the wrapper is transparent.

#### Using vLLM:

```bash
literegistry vllm \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --registry redis://login-node:6379 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

<diagram of GPU node running vLLM: shows the model loaded across 4 GPUs with tensor parallelism, connected to registry with heartbeat arrow. Label the node with "CHECKPOINT NODE - Can be preempted!" to emphasize the dynamic nature>

#### Using SGLang:

```bash
literegistry sglang \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --registry redis://login-node:6379 \
  --tp-size 4 \
  --mem-fraction-static 0.9
```

You can launch multiple instances of the same model across different nodes for load balancing, or run different models—LiteRegistry handles all the tracking automatically.

**Notice something?** The only thing that changed from your normal vLLM/SGLang command is adding `--registry redis://login-node:6379`. That's it. You don't rewrite your launch scripts, you don't containerize anything, you don't change your Slurm submission workflow. You just add one flag and get automatic service discovery.

**Example: Running multiple replicas**

```bash
# On GPU node 1
literegistry vllm --model "meta-llama/Llama-3.1-8B-Instruct" --registry redis://login:6379

# On GPU node 2  
literegistry vllm --model "meta-llama/Llama-3.1-8B-Instruct" --registry redis://login:6379

# On GPU node 3
literegistry vllm --model "mistralai/Mixtral-8x7B-Instruct-v0.1" --registry redis://login:6379
```

All three servers automatically register and start sending heartbeats.

### Step 3: Start the Gateway

Launch the Gateway server to handle client requests:

```bash
literegistry gateway \
  --registry redis://login-node:6379 \
  --host 0.0.0.0 \
  --port 8080
```

The Gateway immediately queries the registry and starts routing traffic to available servers.

<diagram of complete system architecture: multiple vLLM/SGLang servers on different GPU nodes -> central Redis registry <- Gateway server <- multiple clients>

### Step 4: Monitor Your Cluster

Use the CLI to check cluster status:

```bash
literegistry summary --registry redis://login-node:6379
```

Output:
```
meta-llama/Llama-3.1-8B-Instruct: 2
mistralai/Mixtral-8x7B-Instruct-v0.1: 1
```

This shows you have 2 replicas of Llama running and 1 Mixtral instance.

## Using the Gateway API

Once your cluster is running, clients can send requests to the Gateway using the OpenAI-compatible API:

```bash
curl -X POST http://gateway-host:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 100
  }'
```

The Gateway automatically:
1. Looks up available servers for that model
2. Selects the server with the lowest latency
3. Routes the request
4. Returns the response
5. Updates metrics

If that server is down, it automatically tries the next available replica.

## Writing Code: Server-Side Integration

### Registering a Custom Server

If you're not using vLLM or SGLang, you can still register any HTTP service with LiteRegistry:

```python
from literegistry import RegistryClient, get_kvstore
import asyncio

async def register_my_server():
    # Connect to registry (Redis or filesystem)
    store = get_kvstore("redis://localhost:6379")
    # Or for filesystem: store = get_kvstore("/shared/registry")
    
    client = RegistryClient(store, service_type="model_path")
    
    # Register your server
    await client.register(
        port=8000,
        metadata={
            "model_path": "my-custom-model",
            "model_type": "custom-transformer"
        }
    )
    
    print("Server registered! Starting heartbeats...")
    
    # Keep server alive with heartbeats
    while True:
        await asyncio.sleep(10)  # Heartbeat every 10 seconds
        await client.heartbeat(port=8000)

asyncio.run(register_my_server())
```

This pattern lets you integrate any model server into the LiteRegistry ecosystem.

### Querying Available Models

```python
from literegistry import RegistryClient, get_kvstore
import asyncio

async def list_models():
    store = get_kvstore("redis://localhost:6379")
    client = RegistryClient(store, service_type="model_path")
    
    # Get all available models and their servers
    models = await client.models()
    
    for model_name, servers in models.items():
        print(f"\n{model_name}:")
        for server in servers:
            print(f"  - {server['base_url']}")
            print(f"    Last heartbeat: {server.get('last_heartbeat_time')}")
            print(f"    Request stats: {server.get('request_stats', {})}")

asyncio.run(list_models())
```

## Writing Code: Client-Side Usage

### Basic HTTP Client with Automatic Failover

The `RegistryHTTPClient` provides automatic failover and retry logic:

```python
from literegistry import RegistryClient, RegistryHTTPClient, get_kvstore
import asyncio

async def make_request():
    store = get_kvstore("redis://localhost:6379")
    client = RegistryClient(store, service_type="model_path")
    
    # Create HTTP client for a specific model
    async with RegistryHTTPClient(
        client, 
        "meta-llama/Llama-3.1-8B-Instruct"
    ) as http_client:
        
        # Make request with automatic retry and rotation
        result, server_url = await http_client.request_with_rotation(
            endpoint="v1/completions",
            payload={
                "prompt": "Write a haiku about distributed systems",
                "max_tokens": 50
            },
            timeout=30,
            max_retries=3
        )
        
        print(f"Response from {server_url}:")
        print(result)

asyncio.run(make_request())
```

If the first server fails or times out, the client automatically tries the next available replica.

### Batch Processing with Parallel Requests

For high-throughput workloads, process multiple requests in parallel:

```python
async def batch_inference():
    store = get_kvstore("redis://localhost:6379")
    client = RegistryClient(store, service_type="model_path")
    
    # Prepare batch of requests
    prompts = [
        {"prompt": f"Question {i}: Tell me about AI", "max_tokens": 50}
        for i in range(100)
    ]
    
    async with RegistryHTTPClient(
        client,
        "meta-llama/Llama-3.1-8B-Instruct"
    ) as http_client:
        
        # Process 100 requests with max 5 concurrent
        results = await http_client.parallel_requests(
            endpoint="v1/completions",
            payloads_list=prompts,
            max_parallel_requests=5,
            timeout=30,
            max_retries=3
        )
        
        print(f"Processed {len(results)} requests")
        for i, (result, server) in enumerate(results):
            print(f"Request {i} served by {server}")

asyncio.run(batch_inference())
```

The client automatically distributes load across available replicas and handles failures gracefully.

## A Day in the Life: LiteRegistry in a Research Lab

Let's walk through a realistic scenario that shows why LiteRegistry matters:

**9 AM**: You submit a Slurm job to start 4 vLLM servers on your dedicated nodes. LiteRegistry's vLLM integration automatically registers each server as it comes online. Your inference dashboard shows 4 healthy servers.

**11 AM**: Checkpoint nodes become available. You submit another job requesting 12 additional nodes. LiteRegistry detects the new servers and adds them to the pool as they register. Now you have 16 servers handling requests.

**2 PM**: A high-priority job preempts 8 of your checkpoint nodes. Your vLLM servers on those nodes disappear. With traditional tools, this would mean:
- Failed requests from clients still trying those endpoints
- Manual intervention to update your load balancer
- Downtime while you figure out what happened

With LiteRegistry:
- Servers fail to send heartbeats within seconds
- LiteRegistry marks them as unavailable automatically  
- Client requests route to the 8 remaining healthy servers
- Zero downtime, zero manual intervention

**4 PM**: Those checkpoint nodes become available again. You restart your Slurm job. The servers register themselves and start receiving traffic—automatically.

**All day long**: Graduate students across your lab are running inference workloads. They don't need to know which nodes are up or down. They just send requests to LiteRegistry's gateway, and it routes them intelligently to whatever servers are currently available.

<diagram of a timeline showing the scenario above: nodes appearing, disappearing, and reappearing throughout the day, with LiteRegistry automatically managing the fleet. Show a graph of "Available GPUs" fluctuating from 4->16->8->16 throughout the day, with requests being served continuously despite the changes>

This is the power of LiteRegistry: **it turns the chaos of shared HPC resources into a reliable, self-managing service**.

## Storage Backend Trade-offs

Choosing between FileSystem and Redis backends depends on your deployment:

### FileSystem Backend

**Use when:**
- Running on a single machine for development/testing
- All nodes share a filesystem (common in HPC with NFS)
- You want zero additional dependencies
- You have a small, stable deployment (5-10 servers)

**Limitations:**
- Can bottleneck with many concurrent services/clients
- File locking overhead increases with scale
- Not ideal for 50+ services or high query rates
- Can struggle when many nodes are rapidly joining/leaving (e.g., checkpoint nodes being preempted and restarted frequently)

**Example:**
```python
from literegistry import FileSystemKVStore
store = FileSystemKVStore("/shared/cluster/registry")
```

<diagram showing shared filesystem: multiple nodes accessing NFS mount point with file locks and potential contention>

### Redis Backend (Recommended for Production)

**Use when:**
- Running across multiple nodes without shared storage
- Need high-concurrency access (many clients/services)
- Running production workloads with 10+ services
- **Using checkpoint/preemptible nodes that frequently come and go**
- Deploying on cloud infrastructure with spot instances

**Advantages:**
- Built for distributed systems and high concurrency
- Atomic operations without file locking overhead
- Better performance at scale
- Handles rapid node churn (nodes joining/leaving) much better than filesystem
- Native pub/sub for future features
- Battle-tested in production environments

**Why it matters for checkpoint resources:**  
When you're running on a mix of dedicated and checkpoint nodes, Redis shines. It can handle dozens of servers registering, deregistering, sending heartbeats, and updating metrics simultaneously without breaking a sweat. The filesystem backend might struggle when 10 checkpoint nodes all get preempted at once and then 12 new ones register 5 minutes later.

**Example:**
```python
from literegistry import RedisKVStore
store = RedisKVStore("redis://login-node:6379")
```

<diagram showing Redis architecture: central Redis server with multiple concurrent connections from different nodes, showing high-throughput capability. Emphasize rapid registration/deregistration events happening simultaneously>

**Rule of thumb:** 
- **Development/testing**: Filesystem is fine
- **Small stable deployment** (4-8 dedicated nodes): Filesystem works, Redis is better
- **Dynamic environments with checkpoint resources**: Use Redis
- **Production or 10+ services**: Definitely Redis

If you're dealing with the reality of checkpoint nodes getting preempted, save yourself the headache and start with Redis from day one.

## Real-World Example: HPC Research Cluster with Mixed Resources

Here's how a typical research group might deploy LiteRegistry on an HPC cluster with both dedicated and checkpoint resources:

**Setup:**
- 4 dedicated GPU nodes (guaranteed), each with 4x A100 GPUs  
- 6-12 checkpoint GPU nodes (come and go based on cluster load), each with 4x A100 GPUs
- Shared NFS storage for datasets
- Redis running on login node for coordination
- Slurm scheduler managing all allocations

**Deployment:**

```bash
# 1. Start Redis on login node (persistent)
literegistry redis --port 6379

# 2. Submit SLURM job for DEDICATED nodes (partition=dedicated)
# These 4 nodes are guaranteed to stay up
sbatch --partition=dedicated --nodes=4 --gres=gpu:4 --wrap="
  srun literegistry vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --registry redis://login-node:6379 \
    --tensor-parallel-size 4
"

# 3. Submit SLURM job for CHECKPOINT nodes (partition=checkpoint)  
# These can be preempted at any time - that's okay!
sbatch --partition=checkpoint --nodes=8 --gres=gpu:4 --wrap="
  srun literegistry vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --registry redis://login-node:6379 \
    --tensor-parallel-size 4
"

# 4. Start Gateway on login node (persistent)
literegistry gateway \
  --registry redis://login-node:6379 \
  --host 0.0.0.0 \
  --port 8080
```

**What happens in practice:**
- Your 4 dedicated nodes register immediately and stay up
- The 8 checkpoint nodes register as they're allocated (might take a few minutes)
- When checkpoint nodes are preempted, they disappear from the registry automatically
- The gateway keeps routing traffic to whatever nodes are currently available
- When checkpoint nodes become available again, you resubmit the job and they rejoin automatically

**The key insight**: You're not managing 12 individual endpoints. You're managing a *pool* of replicas that grows and shrinks dynamically. LiteRegistry handles the complexity.

Now all researchers can submit requests to `http://login-node:8080`, and the system automatically handles routing, load balancing, and failover.

---

**💡 Real Talk: Why This Matters for Checkpoint Resources**

In this setup, when your checkpoint job gets killed:
1. Those vLLM processes die
2. They stop sending heartbeats to Redis
3. Within 10-30 seconds, LiteRegistry marks them as dead
4. New requests automatically route to your 4 dedicated nodes
5. Your users see maybe 1-2 failed requests during the transition
6. Everything else just works

When checkpoint resources become available again:
1. Resubmit your Slurm job (or use a cron job to auto-resubmit)
2. New vLLM servers start and register themselves
3. Within seconds, they start receiving traffic
4. Load automatically rebalances across all available servers

**Without LiteRegistry**, you'd be manually updating load balancer configs, dealing with hardcoded endpoints, and fielding Slack messages from labmates asking why inference is broken.

**With LiteRegistry**, this entire dance happens automatically. You just resubmit the Slurm job and walk away.

---

<diagram showing full HPC deployment: login node with Redis + Gateway at the center. Show two groups of compute nodes: (1) "DEDICATED PARTITION" with 4 solid/permanent-looking nodes, and (2) "CHECKPOINT PARTITION" with 8 nodes that appear semi-transparent or with dashed borders to show they're temporary. Arrows showing heartbeats from all nodes to registry, and request routing from gateway to nodes. Some checkpoint nodes could be faded/grayed out to show they've been preempted>

## Advanced Features

### Custom Health Checks

You can implement custom health checks for your servers:

```python
async def custom_health_check(server_info):
    """Check if server is truly healthy beyond just heartbeat"""
    # Add custom logic here
    return server_info.get('last_heartbeat_time') is not None
```

### Metrics and Monitoring

LiteRegistry tracks detailed metrics for each server:
- Request counts (last 5/15/60 minutes)
- Latency percentiles (p50, p90, p99)
- Error rates
- Throughput

Access these via the client:

```python
async def get_metrics():
    store = get_kvstore("redis://localhost:6379")
    client = RegistryClient(store, service_type="model_path")
    
    models = await client.models()
    for model_name, servers in models.items():
        for server in servers:
            stats = server.get('request_stats', {})
            print(f"{model_name} @ {server['base_url']}:")
            print(f"  Last 15min requests: {stats.get('last_15_minutes', 0)}")
            print(f"  Avg latency: {stats.get('last_15_minutes_latency', 0):.2f}ms")
```

## Why LiteRegistry? (And Why Not Just Use Kubernetes?)

At this point you might be thinking: "We already have Kubernetes/Consul/Nomad for this."

Yes, these are powerful orchestration platforms. They absolutely can handle service discovery and failover. But consider the cost:

**What Kubernetes requires:**
- Learning container orchestration
- Dockerizing all your ML code
- Setting up and maintaining a Kubernetes cluster (or paying for managed K8s)
- Writing deployment YAML for each service
- Understanding pods, deployments, services, ingresses
- Debugging container networking issues
- Complete infrastructure migration

**What LiteRegistry requires:**
- `pip install literegistry`
- Add one argument to your existing vLLM/SGLang launch command: `--registry redis://your-host:6379`
- That's it

**LiteRegistry is purpose-built for researchers running ML workloads:**
- **Lightweight**: ~500 lines of Python, not a full orchestration platform
- **HPC-native**: Works with Slurm, bare metal, SSH—the tools you already use
- **Model-aware**: Routes based on model names and tracks inference-specific metrics
- **Performance-optimized**: Tracks latency and routes to fastest servers
- **Zero-infrastructure**: No cluster setup, no containers, no infrastructure team required
- **Transparent integration**: Works with standard vLLM/SGLang—all their arguments still work

You can deploy LiteRegistry in 5 minutes on an existing HPC cluster without changing how you submit jobs, how you write code, or how your team works. Try doing that with Kubernetes.

It's the right tool for research teams and ML engineers who need service discovery without becoming DevOps engineers.

## Getting Started Today

LiteRegistry is open source and ready to use:

```bash
# Install
pip install literegistry

# Start serving
literegistry redis --port 6379
literegistry vllm --model meta-llama/Llama-3.1-8B-Instruct --registry redis://localhost:6379
literegistry gateway --registry redis://localhost:6379 --port 8080

# Check status
literegistry summary --registry redis://localhost:6379
```

Whether you're running a single GPU workstation or a 100-node HPC cluster, LiteRegistry provides the coordination layer you need for distributed model inference.

## Getting Started: Try LiteRegistry Today

LiteRegistry is designed to meet you where you are:

**Just exploring?** Install with `pip install literegistry` and try it on your local machine. The filesystem backend works great for development.

**Running a small lab cluster?** Deploy Redis on your login node and start registering your vLLM/SGLang servers. It takes about 5 minutes to get your first distributed setup running.

**Dealing with checkpoint resources and preemption?** This is exactly what LiteRegistry was built for. Let it handle the chaos of nodes coming and going while you focus on your research.

The complete workflow:

```bash
# Install
pip install literegistry

# Start registry (Redis for multi-node, or use filesystem path for single-node)
literegistry redis --port 6379

# Launch your vLLM/SGLang servers (on dedicated or checkpoint nodes)
literegistry vllm --model your-model --registry redis://your-host:6379

# Start the gateway
literegistry gateway --registry redis://your-host:6379 --port 8080

# Check what's running
literegistry summary --registry redis://your-host:6379
```

## Why This Matters

The future of ML research isn't going to be run on perfectly stable, dedicated clusters. It's going to be run on shared resources: checkpoint nodes that come and go, spot instances that can disappear, preemptible VMs that keep costs down.

LiteRegistry is built for this reality. It's built for the graduate student who has 4 dedicated nodes but needs to opportunistically grab 12 more when they're available. It's built for the research lab that can't afford 100% dedicated resources but can make a mixed allocation work. It's built for the real world of modern ML infrastructure.

**Resources:**
- GitHub: [github.com/your-org/literegistry](link to repo)
- Documentation: [docs.literegistry.org](link to docs)
- Examples: [github.com/your-org/literegistry/examples](link to examples)

## Conclusion

If you're tired of babysitting distributed inference deployments, manually updating configs when nodes go down, and explaining to users why their requests are failing, LiteRegistry might be exactly what you need.

It won't make checkpoint resources stop getting preempted. But it will make dealing with that preemption automatic, transparent, and painless.

Try it out on your next project. Whether you're running 4 GPUs or 400, whether your infrastructure is rock-solid or constantly changing, LiteRegistry can help.

---

*Built for researchers who deal with the reality of shared HPC resources. By researchers who've been there.*

*Contributions, issues, and feedback welcome on GitHub!*

