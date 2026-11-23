"""Extract explicitly historical costs and prepare a deterministic fresh GPU pilot."""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manifests/submission_v2/p0'


def stats(values):
    return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
            'p10':float(np.percentile(values,10)),'p90':float(np.percentile(values,90))} if values else {'n':0}


def main():
    out = {'basis':'historical logs, not a new GPU benchmark', 'models':{},
           'fresh_gpu_measurement':'pending restoration of pinned weights and exact dataset inputs'}
    pilot = {'status':'prepared_not_run','scope':'10 legacy development episodes/model; no final-test data',
             'network_downloads_started':False,'models':{},
             'required_before_run':['explicit existing local snapshot path and revision for model/tokenizer',
                                    'pinned HotpotQA source and exact 10-paragraph examples',
                                    'verify selected IDs match question/text stored in legacy archive'],
             'settings':{'nf4':True,'greedy_agent':True,'max_steps':6,'k_samples':5,'sample_temperature':0.7,
                         'sample_batch_size':1,'max_new_tokens':128,'cuda_synchronize_for_each_phase':True,
                         'warmup_episodes':1,'warmup_excluded_from_measurement':True},
             'phase_metrics':['greedy_generation','feature_extraction','alternative_sampling','verbal_prompt',
                              'prefix_replay','total_wall_time','generated_token_count','peak_vram']}
    phase = re.compile(r'^\[(\d+)/\d+ step (\d+)\] (\S+) (.*?) \(([\d.]+)s this episode\)')
    done = re.compile(r'^\[(\d+)/\d+\] (\S+) steps=(\d+) success=(\d+) \(([\d.]+)s/episode,')
    for model in ['qwen7b','llama8b']:
        run=ROOT/'runs'/model
        text=(run/'logs/01_agent.log').read_text(encoding='utf-8',errors='replace')
        pending={};finished={};logged_rate=[]
        for line in text.splitlines():
            if line.startswith('=== Attempt'):
                pending={}
            m=phase.match(line)
            if m:
                _,step,qid,label,elapsed=m.groups()
                if step=='1' and label=='generating action':pending[qid]=[]
                pending.setdefault(qid,[]).append((int(step),label,float(elapsed)))
            m=done.match(line)
            if m:
                _,qid,_,_,rate=m.groups();logged_rate.append(float(rate))
                if pending.get(qid):finished[qid]=pending.pop(qid)
        selected=list(finished.items())[-100:]
        phases=defaultdict(list);bounds=[]
        for _,events in selected:
            bounds.append(events[-1][2])
            for (step,label,t),(nextstep,_,u) in zip(events,events[1:]):
                if u>=t:
                    key={'generating action':'greedy_generation','extracting features':'feature_extraction',
                         'sampling 5 alternatives':'five_alternatives','rating confidence':'verbal_prompt'}.get(label,label)
                    phases[key].append(u-t)
        replay=[]
        for path in (run/'logs').glob('07*.log'):
            replay += [float(x) for x in re.findall(r'\[\d+/\d+\] ([\d.]+)s/episode',path.read_text(encoding='utf-8',errors='replace'))]
        bayes=[float(x) for x in re.findall(r'took (\d+) seconds',(run/'logs/04_trajectory.log').read_text(encoding='utf-8',errors='replace'))]
        out['models'][model]={'last_logged_running_mean_seconds_per_episode':logged_rate[-1] if logged_rate else None,
                             'completed_instrumented_episodes':len(finished),
                             'recent_completed_episode_sample':len(selected),
                             'recent_episode_elapsed_lower_bound_seconds':stats(bounds),
                             'observed_phase_seconds':{k:stats(v) for k,v in phases.items()},
                             'limitations':['Phase timers rounded to 0.1s; final verbal call has no following phase timestamp.',
                                            'Running mean belongs to the corresponding historical collection attempt; includes its historical conditions.',
                                            'Bayesian log entries may include retries; not a current-machine benchmark.'],
                             'last_logged_replay_seconds_per_episode':replay[-1] if replay else None,
                             'logged_four_chain_sampling_seconds':bayes}
        split=json.loads((OUT/'split_manifest.json').read_text())[model]
        pilot['models'][model]={'model_id':split['model'], 'model_revision':None,'tokenizer_revision':None,
                                'local_snapshot_path':None,'dataset_revision':None,'examples_jsonl':None,
                                'episode_ids':split['legacy_development_ids'][:10]}
    base=sum(v['last_logged_running_mean_seconds_per_episode'] for v in out['models'].values())*800/3600
    out['hotpot_1600_new_trajectories_planning_estimate']={
        'episodes_per_model':800,'historical_collection_hours':base,
        'plus_25_percent_scheduling_margin_hours':base*1.25,
        'excluded':['model/data downloads','model load','second environment','human audit','Bayesian refitting'],
        'not_an_empirical_runtime_guarantee':True}
    sampling=[s for v in out['models'].values() for s in v['logged_four_chain_sampling_seconds']]
    out['twenty_bayesian_fits_rough_hours']={'why_twenty':'2 models x 5 outer folds x 2 Bayesian models',
                                          'historical_min_based':min(sampling)*20/3600,
                                          'historical_max_based':max(sampling)*20/3600,
                                          'excludes':'nested probe training, compile overhead, diagnostics and retries'}
    (OUT/'cost_baseline.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    cfg=ROOT/'configs/submission_v2';cfg.mkdir(parents=True,exist_ok=True)
    # Cost recomputation must not erase revisions or paths subsequently supplied.
    if not (cfg/'gpu_pilot.json').exists():
        (cfg/'gpu_pilot.json').write_text(json.dumps(pilot,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out['hotpot_1600_new_trajectories_planning_estimate'],indent=2))


if __name__=='__main__':
    main()
