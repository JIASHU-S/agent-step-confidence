"""Recompute legacy metrics and selected probe fits; never overwrites v1 outputs.

This is a reproduction of the OLD protocol, including target-informed transfer.
It does not implement the corrected P1/P2 experiments or refit Bayesian posteriors.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentconf.metrics import auroc, aurc, ece, platt, summarize
from agentconf.probes import fit_predict, oof_scores
from agentconf.runs import load_run


def read_csv(p):
    with p.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float,np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def feature_matrix(run, steps, folder, key, j):
    by_qid = {}
    for s in steps:
        if s['qid'] not in by_qid:
            with np.load(run / folder / (s['qid']+'.npz')) as z:
                by_qid[s['qid']] = z[key][:,j,:]
    return np.stack([by_qid[s['qid']][s['t']-1] for s in steps]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'results/submission_v2/p0_threads16')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    checks, outputs, times = [], {}, {}

    def check(model, table, row, key, actual, expected, tol=1e-8):
        expected = float(expected)
        actual = float(actual)
        difference = actual-expected
        checks.append({'model':model, 'table':table, 'row':row, 'metric':key,
                       'recomputed':actual, 'archived':expected, 'difference':difference,
                       'tolerance':tol, 'matches':bool(np.isclose(actual,expected,atol=tol,rtol=0,equal_nan=True))})

    with threadpool_limits(limits=args.threads):
        for model in ['qwen7b','llama8b']:
            t0 = time.perf_counter()
            run = ROOT/'runs'/model
            res = run/'results'
            meta, episodes, steps = load_run(run)
            score_map = {(r['qid'],int(r['t'])):r for r in read_csv(res/'step_scores.csv')}
            y = np.array([s['correct'] for s in steps])
            tr = np.array([s['split']=='train' for s in steps]); te = ~tr
            groups = np.array([s['qid'] for s in steps]); kinds = np.array([s['kind'] for s in steps])
            sig = {'probe':np.array([float(score_map[(s['qid'],s['t'])]['probe']) for s in steps])}
            for k in ['lp_mean','lp_min','consistency','verbalized']:
                sig[k] = np.array([s[k] if s[k] is not None else np.nan for s in steps])
            results = {'protocol':'legacy reproduction, NOT corrected confirmatory results', 'step_metrics':[]}
            ref = {(r['signal'],r['kind']):r for r in read_csv(res/'rq1_signals.csv')}
            for name,sc in sig.items():
                prob = sc[te] if name=='probe' else platt(sc[tr],y[tr],sc[te])
                for kind in ['all','Search','Finish']:
                    mask = np.ones(te.sum(),bool) if kind=='all' else kinds[te]==kind
                    raw = np.nan_to_num(sc[te][mask],nan=np.nanmedian(sc[tr]))
                    stat = summarize(raw,prob[mask],y[te][mask])
                    results['step_metrics'].append({'signal':name,'kind':kind,**stat})
                    for k,v in stat.items():
                        check(model,'rq1_signals',name+'/'+kind,k,v,ref[(name,kind)][k])
            print(model,'raw step metrics recomputed',flush=True)
            choice = json.loads((res/'probe_choice.json').read_text())
            j = meta['layers'].index(choice['layer'])
            X = feature_matrix(run,steps,'feats',choice['pooling'],j)
            fitstart = time.perf_counter()
            pred = fit_predict(X[tr],y[tr],X[te])
            results['action_probe_refit'] = {'seconds':time.perf_counter()-fitstart,
                                            'max_probability_abs_diff':float(np.max(abs(pred-sig['probe'][te])))}
            for k,v in summarize(pred,pred,y[te]).items():
                check(model,'selected_probe_refit','action',k,v,ref[('probe','all')][k],tol=1e-6)
            pre = {}
            for r in read_csv(res/'preaction.csv'):
                if r['point'] not in ('pre_step','pre_action'):
                    continue
                F = feature_matrix(run,steps,'feats_pre',r['point'],meta['layers'].index(int(r['layer'])))
                p = fit_predict(F[tr],y[tr],F[te])
                pre[r['point']] = p
                for k,v in summarize(p,p,y[te]).items():
                    check(model,'preaction_refit',r['point'],k,v,r[k],tol=1e-6)
                if r['point']=='pre_step':
                    pre_train = oof_scores(F[tr],y[tr],groups[tr])
                del F
            results['preaction_metrics'] = {k:summarize(v,v,y[te]) for k,v in pre.items()}
            print(model,'selected action/pre-action/pre-step fits completed',flush=True)
            # Refit the position baselines and the stacked pre-step control from stored raw positions.
            t = np.array([s['t'] for s in steps])
            T = np.array([[x==level for level in sorted(set(t))] for x in t],float)
            K = np.array([[x==level for level in sorted(set(kinds))] for x in kinds],float)
            before = np.zeros(len(steps)); before[tr]=pre_train; before[te]=pre['pre_step']
            logit = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
            matrices = {'step index':T,'step index + action type':np.hstack([T,K]),
                        'step index + probe before step':np.column_stack([T,logit(before)]),
                        'step index + action type + probe':np.column_stack([T,K,logit(sig['probe'])])}
            position = {k:LogisticRegression(C=1e4,max_iter=2000).fit(v[tr],y[tr]).predict_proba(v[te])[:,1]
                        for k,v in matrices.items()}
            pref = {r['signal']:r for r in read_csv(res/'position_baseline.csv')}
            for name,p in position.items():
                for k,v in summarize(p,p,y[te]).items():
                    check(model,'position_baseline',name,k,v,pref[name][k],tol=1e-6)

            # Source-domain fit follows the OLD target-informed layer selection for reproduction only.
            with np.load(run/'trivia_feats.npz') as z:
                XS = z[choice['pooling']][:,j,:].astype(np.float32)
            tv = [json.loads(line) for line in (run/'trivia.jsonl').read_text(encoding='utf-8').splitlines() if line]
            ty = np.array([r['correct'] for r in tv])
            trans_test = fit_predict(XS,ty,X[te]); trans_train = fit_predict(XS,ty,X[tr])
            transfer = summarize(trans_test,trans_test,y[te])
            tref = {r['setting']:r for r in read_csv(res/'rq2_transfer.csv')}
            for k,v in transfer.items():
                check(model,'rq2_transfer','legacy zero-shot',k,v,tref['TriviaQA probe, zero-shot'][k],tol=1e-6)
            rng=np.random.default_rng(0); tr_idx=np.flatnonzero(tr)
            recal,small=[],[]
            for rep in range(20):
                pick=rng.choice(tr_idx,size=50,replace=False)
                if len(set(y[pick]))<2:
                    continue
                pos=np.searchsorted(tr_idx,pick)
                recal.append(summarize(trans_test,platt(trans_train[pos],y[pick],trans_test),y[te]))
                ps=fit_predict(X[pick],y[pick],X[te],n_components=49)
                small.append(summarize(ps,ps,y[te]))
            results['transfer']={'legacy_zero_shot':transfer,'source_accuracy':float(ty.mean())}
            for name,values in [('TriviaQA probe + Platt on 50 agent steps',recal),('in-domain probe on 50 agent steps',small)]:
                row={}
                for k in ['auroc','brier','ece']:
                    vals=np.array([r[k] for r in values])
                    row[k]=float(vals.mean());row[k+'_sd']=float(vals.std())
                results['transfer'][name]=row
                for k,v in row.items():
                    check(model,'rq2_transfer',name,k,v,tref[name][k],tol=1e-6)
            del XS,X
            print(model,'legacy transfer fits completed',flush=True)

            # Whole-episode bootstrap reproduces Table 1 CIs, not the old IID step bootstrap.
            ids=np.unique(groups[te]);members={g:np.flatnonzero(groups[te]==g) for g in ids}
            rng=np.random.default_rng(0)
            baselines={**{k:np.nan_to_num(v[te]) for k,v in sig.items() if k!='probe'},**pre,
                       **{k:v for k,v in position.items() if k in ('step index','step index + action type')}}
            boot={k:[] for k in baselines}
            for _ in range(args.n_boot):
                idx=np.concatenate([members[g] for g in rng.choice(ids,len(ids))])
                a=auroc(sig['probe'][te][idx],y[te][idx])
                for k,v in baselines.items():
                    boot[k].append(auroc(v[idx],y[te][idx])-a)
            results['table1_gaps']={k:{'point':auroc(v,y[te])-auroc(sig['probe'][te],y[te]),
                                      'bootstrap_mean':float(np.mean(boot[k])),
                                      'ci_low':float(np.percentile(boot[k],2.5)),
                                      'ci_high':float(np.percentile(boot[k],97.5))} for k,v in baselines.items()}
            bref={r['comparison']:r for r in read_csv(res/'robust_rq1_bootstrap.csv')}
            for k in ['lp_mean','lp_min','consistency','verbalized']:
                v=results['table1_gaps'][k];r=bref['probe - '+k]
                for key,value,expected in [('bootstrap_mean',v['bootstrap_mean'],-float(r['delta_auroc'])),
                                          ('ci_low',v['ci_low'],-float(r['ci_high'])),('ci_high',v['ci_high'],-float(r['ci_low']))]:
                    check(model,'episode_bootstrap',k,key,value,expected)

            # Re-evaluate the archived Bayesian predictions, without claiming to refit their posterior.
            ep=read_csv(res/'rq3_episodes.csv');sy=np.array([int(r['success']) for r in ep])
            curves={k:np.array([float(r[k]) for r in ep]) for k in ['M0_last','M0_min','M0_product','M1_forward']}
            tre={r['method']:r for r in read_csv(res/'rq3_trajectory.csv')}
            results['trajectory_metrics']={k:summarize(v,v,sy) for k,v in curves.items()}
            for name,stat in results['trajectory_metrics'].items():
                for k,v in stat.items():check(model,'rq3_trajectory',name,k,v,tre[name][k])
            m0=np.array([min(json.loads(r['running_M0'])) for r in ep]);m1=np.array([min(json.loads(r['running_M1'])) for r in ep])
            rng=np.random.default_rng(1)
            boot3={k:[] for k in ['aurc_M0_minus_M1','ece_product_minus_forward','auroc_forward_minus_last','ece_last_minus_forward']}
            for _ in range(args.n_boot):
                i=rng.integers(0,len(sy),len(sy))
                boot3['aurc_M0_minus_M1'].append(aurc(m0[i],sy[i])-aurc(m1[i],sy[i]))
                boot3['ece_product_minus_forward'].append(ece(curves['M0_product'][i],sy[i])-ece(curves['M1_forward'][i],sy[i]))
                boot3['auroc_forward_minus_last'].append(auroc(curves['M1_forward'][i],sy[i])-auroc(curves['M0_last'][i],sy[i]))
                boot3['ece_last_minus_forward'].append(ece(curves['M0_last'][i],sy[i])-ece(curves['M1_forward'][i],sy[i]))
            results['trajectory_bootstrap']={k:{'mean':float(np.mean(v)),'ci_low':float(np.percentile(v,2.5)),
                                                 'ci_high':float(np.percentile(v,97.5))} for k,v in boot3.items()}
            rr=json.loads((res/'robust_rq3.json').read_text());pr=json.loads((res/'position_rq3.json').read_text())
            for name,stat in results['trajectory_bootstrap'].items():
                expected=pr[name] if name=='ece_last_minus_forward' else rr[name]
                for k,v in stat.items():check(model,'trajectory_bootstrap',name,k,v,expected[k])
            results['ece_gap_reporting']={
                'product_minus_forward_point':ece(curves['M0_product'],sy)-ece(curves['M1_forward'],sy),
                'product_minus_forward_bootstrap_mean':results['trajectory_bootstrap']['ece_product_minus_forward']['mean'],
                'last_minus_forward_point':ece(curves['M0_last'],sy)-ece(curves['M1_forward'],sy)}
            results['not_recomputed']=['Full layer/pooling search OOF grid','Bayesian posterior and original PSIS-LOO',
                                      'Full plotted hand-off confidence bands','Lenient-label refit']
            outputs[model]=results;times[model]=time.perf_counter()-t0
            write_json(args.out/(model+'_recomputed.json'),results)
            print(model,f'completed in {times[model]:.1f}s',flush=True)
    write_json(args.out/'reproduction_summary.json',{'comparison_count':len(checks),
               'matches':sum(r['matches'] for r in checks),'mismatches':[r for r in checks if not r['matches']],
               'seconds_by_model':times,'threads':args.threads,'bootstrap_replicates':args.n_boot,
               'scope':'Archived inputs, selected probe fits, scalar metrics and selected paired intervals; not full end-to-end generation.'})
    with (args.out/'metric_checks.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    print(f'{sum(r["matches"] for r in checks)}/{len(checks)} comparisons matched',flush=True)
    return int(any(not r['matches'] for r in checks))


if __name__=='__main__':
    raise SystemExit(main())
