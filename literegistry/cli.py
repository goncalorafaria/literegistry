
import pprint
from termcolor import colored
import asyncio
from literegistry import RegistryClient, get_kvstore
import fire
import literegistry.redis as redis
import literegistry.gateway as gateway
import literegistry.vllm_wrapper as vllm
import literegistry.sglang_wrapper as sglang
    
def check_registry(verbose=False, registry_dir="/gscratch/ark/graf/registry"):
    
    r = RegistryClient(get_kvstore(registry_dir))
    #r = RegistryClient(FileSystemKVStore("/gscratch/ark/graf/registry"))

    pp = pprint.PrettyPrinter(indent=1, compact=True)

    for k, v in asyncio.run(r.models()).items():
        print(f"{colored(k, 'red')}")
        for item in v:
            print(colored("--" * 20, "blue"))
            for key, value in item.items():

                if key == "request_stats":
                    if verbose:
                        print(f"\t{colored(key, 'green')}:{value}")
                    else:
                        if "last_15_minutes_latency" in value:
                            nvalue = value["last_15_minutes"]
                            print(f"\t{colored(key, 'green')}:{colored(nvalue,'red')}")
                        else:
                            print(f"\t{colored(key, 'green')}:NO METRICS YET.")
                else:
                    print(f"\t{colored(key, 'green')}:{value}")

    # pp.pprint(r.get("allenai/Llama-3.1-Tulu-3-8B-SFT"))


def check_summary(registry="redis://klone-login01.hyak.local:6379"):
    r = RegistryClient(get_kvstore(registry))

    for k, v in asyncio.run(r.models()).items():
        print(f"{colored(k, 'red')} :{colored(len(v),'green')}")


def main():
    
    fire.Fire({
        "summary": check_summary,
        "redis": redis.main,
        "gateway": gateway.main,
        "vllm": vllm.main,
        "sglang": sglang.main,
    })


if __name__ == "__main__":

    main()