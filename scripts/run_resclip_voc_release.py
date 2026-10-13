#!/usr/bin/env python3
"""Paired full VOC evaluation; use the same runtime for both window settings."""
import fcntl,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/rebuttal/resclip_voc_release_20261013'
def save(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def main():
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for label,extra in [('baseline_replay',['--baseline-replay','--job-dir',str(OUT/'baseline_replay')]),('release_correction',[])]:
  command=['bash',str(OUT/'code/python_e4'),str(OUT/'code/measure_resclip_voc_release.py'),'--run',str(OUT),'--method','resclip','--dataset','voc20',*extra]
  with (OUT/(label+'_full.log')).open('a') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   save('queue_status.json',dict(status='running',stage=label,worker_pid=child.pid,pid=__import__('os').getpid()))
   if child.wait()!=0:
    save('queue_status.json',dict(status='failed',stage=label));raise RuntimeError(label+' failed')
 save('queue_status.json',dict(status='inference_complete',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
if __name__=='__main__':main()
