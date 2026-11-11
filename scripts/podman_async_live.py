import asyncio,json,pathlib,time,collections,statistics,sys,argparse
from literegistry_tool_client.podman_async import PodmanAsyncClient
from literegistry_tool_client.podman import PodmanStreamError
parser=argparse.ArgumentParser(description="Exercise an explicitly supplied streaming Podman fleet")
parser.add_argument("mode", choices=["smoke", "load"])
parser.add_argument("--gateway", required=True)
parser.add_argument("--output-dir", type=pathlib.Path, required=True)
parser.add_argument("--waves", type=int, default=6)
parser.add_argument("--sessions", type=int, default=128)
parser.add_argument("--steps", type=int, default=6)
args=parser.parse_args()
if min(args.waves,args.sessions,args.steps)<1:parser.error("waves, sessions and steps must be positive")
R=args.output_dir;R.mkdir(parents=True,exist_ok=True)
G=args.gateway.rstrip("/");MODE=args.mode
LOG=(R/(MODE+'-events.jsonl')).open('w',buffering=1)
counts=collections.Counter();latencies=[];containers=[]
def record(kind,**data):
 counts[kind]+=1;LOG.write(json.dumps({'time':time.time(),'kind':kind,**data})+'\n')
def client(id):return PodmanAsyncClient(G,image='public.ecr.aws/docker/library/python:3.12-slim',workdir='/tmp',client_id='stream32-'+id,timeout=90,handshake_timeout=600)
async def episode(wave,i,steps):
 c=client(f'{wave}-{i}');phase='open'
 try:
  began=time.monotonic();await c.open();containers.append(c.container_id);record('opened',wave=wave,id=i,cid=c.container_id,seconds=time.monotonic()-began)
  for step in range(steps):
   phase='command';token=f'{wave}:{i}:{step}';start=time.monotonic()
   cmd=f"printf '{token}:A\\n'; sleep 0.03; printf '{token}:E\\n' >&2; printf '{token}:B\\n'"
   job=await c.start(command=cmd,timeout=20);start_seconds=time.monotonic()-start;out='';err=''
   if step%3==0:
    async for b in job.batches(wait_ms=1000):out+=b['stdout'];err+=b['stderr']
    result=await job.wait()
   elif step%3==1:
    b=await job.read();out+=b['stdout'];err+=b['stderr'];result=await job.wait()
   else:result=await job.wait()
   out+=result['stdout'];err+=result['stderr']
   assert out==f'{token}:A\n{token}:B\n',(out,token)
   assert err==f'{token}:E\n',(err,token)
   assert result['exit_code']==0 and result['success'],result
   again=await job.read();assert again['stdout']==again['stderr']=='' and again['done'],again
   duration=time.monotonic()-start;latencies.append(duration);record('command_ok',wave=wave,id=i,step=step,seconds=duration,start_seconds=start_seconds)
 except Exception as e:record('error',wave=wave,id=i,phase=phase,type=type(e).__name__,code=getattr(e,'code',None),error=str(e))
 finally:
  try:await c.close();record('closed',wave=wave,id=i)
  except Exception as e:record('cleanup_error',wave=wave,id=i,error=str(e))
async def edges():
 async with client('edges') as c:
  job=await c.start(command='printf early; sleep 2; printf late',timeout=10)
  b=await job.read(wait_ms=1000)
  while not b['stdout'] and not b['done']:b=await job.read(wait_ms=1000)
  assert b['stdout']=='early' and b['running'],b
  try:await c.start(command='echo forbidden',timeout=5)
  except PodmanStreamError as e:assert e.code=='session_busy',str(e);record('busy_rejected',code=e.code)
  else:raise AssertionError('concurrent command accepted')
  z=await job.wait();assert z['stdout']=='late' and z['success'];record('incremental_before_exit_ok')
  j=await c.start(command='printf nope >&2; exit 7');z=await j.wait();assert z['exit_code']==7 and not z['success'] and z['stderr']=='nope',z;record('nonzero_exit_ok')
  j=await c.start(command='sleep 3',timeout=0.3);z=await j.wait();assert z['timed_out'] and not z['success'],z;record('timeout_ok')
  j=await c.start(command="python3 -c 'print(\"é漢🙂\"*10000)'",timeout=20);z=await j.wait(max_output_chars=100);assert z['output_truncated'] and len(z['stdout'])==100 and z['dropped_output_chars']==29901,z;record('bounded_unicode_output_ok')
  j=await c.start(command='printf out; printf err >&2',merge_output=True);z=await j.wait();assert z['stdout']=='outerr' and z['stderr']=='',z;record('merged_output_ok')
async def concurrency_edges():
 async with client('concurrent-start') as c:
  results=await asyncio.gather(*(c.start(command=f'printf race{k}; sleep 2',timeout=10) for k in range(8)),return_exceptions=True)
  accepted=[x for x in results if not isinstance(x,BaseException)]
  rejected=[x for x in results if isinstance(x,PodmanStreamError) and x.code=='session_busy']
  assert len(accepted)==1 and len(rejected)==7,[str(x) for x in results]
  await accepted[0].wait();record('concurrent_starts_ok',accepted=1,busy=7)
  job=await c.start(command="for i in $(seq 1 40); do printf '%s\\n' \"$i\"; sleep 0.01; done",timeout=15)
  pieces=[]
  async def reader():
   while not job.done:
    batch=await job.read(wait_ms=1000)
    pieces.extend(batch['stdout'].splitlines())
  await asyncio.gather(*(reader() for _ in range(4)))
  await job.wait()
  assert sorted(map(int,pieces))==list(range(1,41)),pieces
  record('concurrent_readers_ok',readers=4,lines=len(pieces))
 c=client('close-running');await c.open()
 job=await c.start(command='printf before; sleep 1; printf after',timeout=10)
 began=time.monotonic();await c.close()
 assert not c.opened and job.done
 record('close_running_ok',seconds=time.monotonic()-began)

async def main():
 t=time.monotonic()
 if MODE=='smoke':
  try:
   await edges()
   await concurrency_edges()
  except Exception as e:record('edge_error',type=type(e).__name__,error=str(e))
  await episode(0,0,12)
 else:
  for wave in range(args.waves):
   await asyncio.gather(*(episode(wave,i,args.steps) for i in range(args.sessions)))
   print('WAVE',wave,dict(counts),flush=True)
 elapsed=time.monotonic()-t;ordered=sorted(latencies)
 summary={'mode':MODE,'counts':dict(counts),'elapsed_seconds':elapsed,'commands_per_second':len(latencies)/elapsed,'latency_p50':statistics.median(latencies) if latencies else None,'latency_p95':ordered[int(.95*(len(ordered)-1))] if ordered else None,'containers':len(containers),'gateway':G}
 (R/(MODE+'-summary.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
asyncio.run(main())
sys.exit(1 if any(counts[k] for k in ("error", "edge_error", "cleanup_error")) else 0)
