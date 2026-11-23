"""Opt-in CRiSH-G v2 controller; legacy local/global paths remain separate."""
import copy
from contextlib import nullcontext
import json
import math
import time
import numpy as np
import torch
from .engine import _check_deadline,_draw_context,cached_views,fixed_modes,projector_optimizer
from .evidence_v2 import evidence_for_epoch
from .global_policy import epoch_arrays
from .lookahead import (clone_snapshot,preserve_rng,assert_plain_sgd,component_gradients_after_keep,
                        first_order_component_gains)
from .normalized_selector import component_statistics,select_dose,failures,COMPONENTS
from .probe import LegalProbe,ProbeWeights
from .rank_policy import apply_rank_dose,ema_scores,hash_key


class RankPopulationController:
    def __init__(self,owner):
        self.owner=owner; self.runtime=owner.runtime; self.settings=owner.settings
        self.probe_settings=owner.probe_settings
        saved=getattr(self.runtime,'restored_population_policy',None) or {}
        self.policy=dict(q=float(saved.get('q',0.)),selected_epoch=saved.get('selected_epoch'),
                         sigma_history=copy.deepcopy(saved.get('sigma_history',{c:[] for c in COMPONENTS})),
                         ema=copy.deepcopy(saved.get('ema',{})))
        owner.policy=self.policy
        if self.settings.diagnostic_oracle and 'oracle' not in self.runtime.run_dir.name.lower():
            raise ValueError('Diagnostic oracle run directory must contain oracle')

    def save_policy(self):
        self.runtime.checkpoint_extra=dict(getattr(self.runtime,'checkpoint_extra',{}),
                                           population_policy_state=copy.deepcopy(self.policy))

    def apply(self,state,evidence,q,scores,epoch):
        return apply_rank_dose(state,evidence,q,self.settings,seed=self.runtime.config['seed'],epoch=epoch,
                              scores=scores,min_support=self.probe_settings.min_support,
                              max_protected_shift=self.probe_settings.max_protected_shift)

    def prepare(self,state,model,head,optimizer,dataset,args,epoch):
        s=self.settings;cfg=self.runtime.config
        if cfg['method']!='crish' or not s.enabled or epoch<s.warmup_epochs:return state
        started=time.monotonic();due=s.due(epoch);deadline=started+s.emergency_deadline_seconds if due else None
        source='refresh' if due else 'persisted' if s.persist else 'nopersist_zero'
        record=dict(epoch=epoch,profile=cfg.get('experiment_profile','g2'),status='completed',
                    deadline_triggered=False,source=source,q_grid=list(s.dose_quantiles),candidates=[],
                    notices=[],oracle_diagnostic=s.diagnostic_oracle,q_hat=0.,q_candidate=0.,
                    gradient_seconds=0.,scoring_seconds=0.,policy_build_seconds=0.)
        evidence=single=scores=coverage=None;outcome=state;q=0.;resets=[]
        device=next(head.parameters()).device
        try:
            with preserve_rng(),(fixed_modes(model) if model is not None else nullcontext()):
                # Private refresh seed controls augmentations; preserve_rng restores all global streams.
                if due:
                    import random
                    seed=hash_key(cfg['seed'],epoch)%(2**32)
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                evidence,single=evidence_for_epoch(state,model,dataset,s,device=device,seed=cfg['seed'],epoch=epoch,
                    batch_size=s.train_batch,num_workers=cfg.get('num_workers',0),deadline=deadline)
                scores,ema,resets=ema_scores(state,evidence,self.policy['ema'],s)
                self.policy['ema']=ema
                record['evidence_seconds']=time.monotonic()-started
                record['ranking']='two_view_reciprocal_agreement' if s.evidence_views==2 else 'single_view_reciprocal_agreement'
                if due:
                    if s.gate=='none':
                        q=s.fixed_dose
                    else:
                        q=self.select(state,evidence,scores,model,head,optimizer,dataset,args,epoch,deadline,record)
                    _check_deadline(deadline)
                    self.policy.update(q=float(q),selected_epoch=epoch)
                else:q=self.policy['q'] if s.persist else 0.
                before_apply=time.monotonic()
                outcome,coverage=self.apply(state,evidence,q,scores,epoch)
                record['policy_seconds']=time.monotonic()-before_apply
                if due:_check_deadline(deadline)
        except (ValueError,RuntimeError) as error:
            outcome=state;q=0.;coverage=None
            record.update(status='fallback_keep',reason=str(error),deadline_triggered=(deadline is not None and time.monotonic()>=deadline))
            if due:self.policy.update(q=0.,selected_epoch=epoch)
        finally:
            self.save_policy()
            record.update(q=float(q),chosen_threshold=float(q),chosen_fraction=float(q),
                          accepted=int(outcome is not state),seconds=time.monotonic()-started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                          peak_memory_scope='epoch_including_refresh',ema_resets=resets)
            levels=(coverage or {}).get('levels',[])
            record['abstained_unlabeled']=[x['abstained_unlabeled'] for x in levels]
            record['moved']=(coverage or {}).get('moved',0)
            epoch_record=dict(epoch=epoch,q=float(q),tau=float(q),source=source,status=record['status'],
                ranking=record.get('ranking'),levels=levels,moved=(coverage or {}).get('moved',0),ema_resets=resets,
                seconds=record['seconds'],knn_seconds=record.get('evidence_seconds'),oracle_diagnostic=s.diagnostic_oracle)
            if due:
                self.runtime.last_population_record=record
                self.write('crish_p.jsonl',record)
            self.runtime.last_policy_epoch_record=epoch_record
            self.write('crish_policy_epochs.jsonl',epoch_record)
            if s.save_arrays and evidence is not None and (due or epoch%5==0):
                arrays=epoch_arrays(state,outcome,evidence)
                # v2 diagnostics retain full score precision for AUROC.
                for l,r in enumerate(evidence.reliability):
                    arrays[f'reliability_{l}']=r.numpy()
                    arrays[f'single_view_reliability_{l}']=single.reliability[l].numpy()
                    if scores is not None and (s.reliability_ema or {}).get('enabled',False):
                        arrays[f'ema_reliability_{l}']=scores[l].numpy()
                np.savez_compressed(self.runtime.run_dir/f'crish_g_epoch_{epoch:04d}.npz',**arrays)
            print(f'CRISH_G_V2 epoch={epoch} q={q} source={source} status={record["status"]}',flush=True)
        return outcome

    def write(self,name,record):
        with (self.runtime.run_dir/name).open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')

    def chunks(self,state,rows,model,dataset,device,deadline):
        views=cached_views(model,dataset,state.ids[rows],device,deadline,batch_size=self.settings.train_batch)
        out=[];size=self.settings.train_batch
        for start in range(0,len(rows),size):
            part=rows[start:start+size];v=views[start:start+size];mask=state.labeled[part]
            out.append(dict(ids=state.ids[part],membership=mask,targets=state.labels[0][part][mask],
                            flat=torch.cat([v[:,0],v[:,1]]).detach()))
        return out

    def draw_chunks(self,state,available,batches,generator):
        n=int(state.labeled.sum())
        weights=torch.where(state.labeled[available],1.,n/len(state.ids)).double()
        if not len(available) or batches<1:raise ValueError('Insufficient disjoint scoring support')
        # Exact train-sampler relative weights, with replacement within each fold.
        # Confirmation excludes every discovery ID, but need not exclude its own repeats.
        return available[torch.multinomial(weights,batches*self.settings.train_batch,replacement=True,generator=generator)]

    def select(self,state,evidence,scores,model,head,optimizer,dataset,args,epoch,deadline,record):
        s=self.settings;cfg=self.runtime.config
        assert_plain_sgd(optimizer)
        if s.probe_float64:record['notices'].append('probe_float64 ignored: fp32 head, fp64 dot accumulation')
        record['notices'].append('normalized_v1 ignores scalar probe component weights')
        generator=torch.Generator().manual_seed(hash_key(cfg['seed'],epoch,'chunks')%(2**63))
        folds,_=_draw_context(state,self.probe_settings,generator,deadline,fold_count=2)
        reserved=torch.cat([torch.cat([k,u]) for k,u in folds])
        available=torch.ones(len(state.ids),dtype=torch.bool);available[reserved]=False
        discovery_rows=self.draw_chunks(state,available.nonzero().flatten(),s.scoring_batches,generator)
        remaining=available.clone();remaining[discovery_rows]=False
        confirmation_rows=(self.draw_chunks(state,remaining.nonzero().flatten(),s.confirmation_batches,generator)
                           if s.confirm and s.confirm_fresh_chunks else discovery_rows)
        record['scoring_ids']=state.ids[discovery_rows].tolist()
        record['confirmation_scoring_ids']=state.ids[confirmation_rows].tolist() if s.confirm else []
        record['probe_ids']=[state.ids[torch.cat([k,u])].tolist() for k,u in folds]
        chunks=self.chunks(state,discovery_rows,model,dataset,next(head.parameters()).device,deadline)
        # The origin clone isolates weight-norm derived tensors and model modes.
        frozen,opt=clone_snapshot(head,projector_optimizer(head,optimizer));frozen.eval()
        parameters=tuple(p for p in frozen.parameters() if p.requires_grad)
        rates={id(p):group['lr'] for group in opt.param_groups for p in group['params']}
        lr=[rates[id(p)] for p in parameters]
        def grads(chunk,current):
            return torch.autograd.grad(self.owner.objective(frozen,chunk['flat'],chunk['ids'],chunk['membership'],
                                       chunk['targets'],current,args,epoch)['loss'],parameters,allow_unused=True)
        gradient_start=time.monotonic()
        g0s=[grads(c,state) for c in chunks]
        record['gradient_seconds']+=time.monotonic()-gradient_start
        g0=tuple(None if all(g[i] is None for g in g0s) else sum(g[i] for g in g0s if g[i] is not None)/len(g0s)
                 for i in range(len(parameters)))
        all_scoring=torch.cat([state.ids[discovery_rows],state.ids[confirmation_rows]])
        registry=dict(zip(state.ids[state.labeled].tolist(),state.known_targets.tolist()))
        probes=[]
        for k,u in (folds if s.confirm else folds[:1]):
            x=cached_views(model,dataset,state.ids[torch.cat([k,u])],next(head.parameters()).device,deadline)
            with torch.no_grad():teacher=frozen(x[len(k):].flatten(0,1)).reshape(len(u),2,-1)
            probes.append(LegalProbe(known_ids=state.ids[k],known_views=x[:len(k)],known_targets=state.labels[0][k],
                unlabeled_ids=state.ids[u],unlabeled_views=x[len(k):],teacher_views=teacher,proposal_ids=all_scoring,
                legal_labels=registry,weights=ProbeWeights(1.,1.,.1,self.probe_settings.probe_std_floor,
                                                           self.probe_settings.probe_view_reduction)))
        if s.confirm:probes[0].assert_disjoint(probes[1])
        before_gradient=time.monotonic()
        query=component_gradients_after_keep(frozen,opt,g0,probes)
        record['probe_gradient_seconds']=time.monotonic()-before_gradient
        record['gradient_seconds']+=record['probe_gradient_seconds']
        from selex_runtime.objective import abstain_delta_loss
        def score(current,parts,fold,keeps=None):
            gains=[];norms=[]
            for m,chunk in enumerate(parts):
                _check_deadline(deadline)
                gradient_start=time.monotonic()
                if all(torch.equal(a,b) for a,b in zip(state.labels,current.labels)):
                    delta=torch.autograd.grad(abstain_delta_loss(frozen,chunk,state,current,args),parameters,allow_unused=True)
                else:
                    keep=keeps[m] if keeps is not None else grads(chunk,state)
                    action=grads(chunk,current)
                    delta=tuple(None if a is None and b is None else
                        (torch.zeros_like(p) if a is None else a).double()-(torch.zeros_like(p) if b is None else b).double()
                        for p,a,b in zip(parameters,action,keep))
                    n0=sum(float(g.double().square().sum()) for g in keep if g is not None)**.5
                    nd=sum(float(g.double().square().sum()) for g in delta if g is not None)**.5
                    norms.append(nd/n0 if n0 else None)
                record['gradient_seconds']+=time.monotonic()-gradient_start
                dot_start=time.monotonic()
                gains.append(first_order_component_gains(query[fold],delta,lr))
                record['scoring_seconds']+=time.monotonic()-dot_start
            return gains,norms
        candidates={};states={};entries={}
        before_score=time.monotonic()
        for q in s.dose_quantiles:
            if q==0:continue
            build_start=time.monotonic()
            current,coverage=self.apply(state,evidence,q,scores,epoch)
            record['policy_build_seconds']+=time.monotonic()-build_start
            entry=dict(q=q,coverage=coverage,constructed=current is not state,status='rejected')
            record['candidates'].append(entry);entries[q]=entry
            if current is state:entry['reason']='empty_policy';continue
            raw,norms=score(current,chunks,0,g0s)
            candidates[q]=raw;states[q]=current
            entry['delta_gradient_relative_norm']=norms
        statistics,scales=component_statistics(candidates,self.policy['sigma_history'],floor_ratio=s.sigma_floor_ratio)
        q,highest=select_dose(statistics,t_min=s.t_min,guard_se=s.guard_se,known_guard=s.known_guard)
        record.update(scales=scales,q_hat=highest,q_candidate=q)
        for dose,stats in statistics.items():
            entries[dose].update(discovery=stats,status='eligible' if not stats['failures'] else 'rejected')
        confirmation_scales=None
        if q and s.confirm:
            parts=(self.chunks(state,confirmation_rows,model,dataset,next(head.parameters()).device,deadline)
                   if s.confirm_fresh_chunks else chunks)
            raw,_=score(states[q],parts,1)
            stats,confirmation_scales=component_statistics({q:raw},self.policy['sigma_history'],floor_ratio=s.sigma_floor_ratio)
            stats[q]['failures']=failures(stats[q],view_min=s.t_min_confirm,guard_se=s.guard_se,
                                         known_guard=s.known_guard,confirmation=True)
            record['confirmation']=dict(q=q,statistics=stats[q],scales=confirmation_scales)
            entries[q]['confirmation']=record['confirmation']
            if stats[q]['failures']:
                entries[q].update(status='rejected',reason='confirmation_failed_no_runner_up');q=0.
        if q:entries[q]['status']='accepted'
        # Update history only after discovery AND confirmation used prior refresh history.
        for c in COMPONENTS:
            if scales:self.policy['sigma_history'][c].append(scales[c]['raw'])
        record['gradient_scoring_seconds']=time.monotonic()-before_score
        return q
