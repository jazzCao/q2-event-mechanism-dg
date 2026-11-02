"""Full two-stage source-only model engine, with explicit attempt authorization.

Not a standalone benchmark runner. An outer runner must bind verified source
cache/splits, budgets and full-panel freeze before any target evaluation.
No authorization or implicit fallback is created by this module.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mocd_full_matching import BlockedIndexedMatching, FullSchedule
from mocd_matching_state import capture_packet, restore_packet
from mocd_paper_components import causal_consistency, matching_contrast, redundancy, weighted_total
from mocd_schedule_control import ScheduleControl
from qualify_ress_literature_components import PaperMOCD


@dataclass
class SourceWindows:
    domain: str
    role: str
    x: torch.Tensor
    y: np.ndarray
    groups: np.ndarray
    indices: np.ndarray

    def validate(self, classes):
        n=len(self.y)
        if self.role not in ("train","val") or not n or self.x.ndim!=2 or self.x.shape!=(n,2048):
            raise ValueError("source window shape/role")
        if self.x.dtype!=torch.float32 or not torch.isfinite(self.x).all():
            raise ValueError("finite FP32 logFFT source input required")
        if self.y.dtype.kind not in "iu" or self.y.shape!=(n,) or np.any(self.y<0) or np.any(self.y>=classes):
            raise ValueError("source labels")
        if self.groups.shape!=(n,) or self.indices.shape!=(n,) or self.indices.dtype.kind not in "iu" or np.any(self.indices<0):
            raise ValueError("source role provenance")
        if len(np.unique(self.indices))!=n:
            raise ValueError("duplicate original window indices")
        if any(t in str(g) for g in self.groups for t in ("K006","KA08","KA09")):
            raise ValueError("protected bearing in source role")


class BoundSources:
    def __init__(self,allowed,forbidden,train,val,classes,data_fingerprint):
        self.allowed=tuple(allowed);self.forbidden=set(forbidden)
        if type(classes) is not int or classes<2:raise ValueError("at least two source classes required")
        if len(self.allowed)<2 or len(set(self.allowed))!=len(self.allowed) or set(self.allowed)&self.forbidden:
            raise ValueError("source-target identity overlap")
        if not data_fingerprint or set(train)!=set(self.allowed) or set(val)!=set(self.allowed):
            raise ValueError("source data binding/coverage")
        self.train=train;self.val=val;self.classes=classes;self.fingerprint=data_fingerprint
        for domain in self.allowed:
            tr,va=train[domain],val[domain]
            tr.validate(classes);va.validate(classes)
            if tr.domain!=domain or va.domain!=domain or tr.role!="train" or va.role!="val":
                raise ValueError("domain role substitution")
            if set(tr.groups.tolist())&set(va.groups.tolist()) or set(tr.indices.tolist())&set(va.indices.tolist()):
                raise ValueError("source train/val leakage")
            if set(tr.y.tolist())!=set(range(classes)):
                raise ValueError("missing real source training class; no borrowing")

    def get(self,domain,role):
        if domain in self.forbidden or domain not in self.allowed:
            raise PermissionError("source sentinel refused domain")
        if role not in ("train","val"):
            raise PermissionError("only source training/validation roles exist")
        return (self.train if role=="train" else self.val)[domain]


def stage_parameters(model,stage):
    if stage not in ("A","B"):raise ValueError("stage")
    names=("gf","gd","pf","pd") if stage=="A" else ("gf","gd","hf","hd","dec")
    return [p for name in names for p in getattr(model,name).parameters()]


class FullModelEngine:
    def __init__(self,sources,profile_name,profiles,fingerprint,device="cpu",schedule=None,permit=None):
        if profile_name not in ("EQ_LITERAL","PROSE_STOPGRAD_SENSITIVITY") or profile_name not in profiles["profiles"] or not fingerprint:
            raise ValueError("unknown profile/unbound engine")
        common=profiles["common"]
        if (common["tau"],common["alpha"],common["lambda_diag"],common["lambda_c"],common["lambda_n"],common["lambda_cross"])!=(.2,.5,1.,.1,.1,.8):
            raise ValueError("profile constants disagree with qualified equation core")
        self.sources=sources;self.profile_name=profile_name
        self.profile={"name":profile_name,"spec":copy.deepcopy(profiles),"matching_backend":"blocked_refined_all_candidates"}
        self.fingerprint=fingerprint;self.device=torch.device(device);self.permit=permit
        self.model=PaperMOCD(sources.classes,len(sources.allowed)).to(self.device)
        self.matching=BlockedIndexedMatching([sources.get(d,"train").y for d in sources.allowed],0)
        self.controller=ScheduleControl(fingerprint,schedule or FullSchedule())
        self.rng=np.random.RandomState(0);self.draws=0;self.stage="A"
        self.optimizers={"A":torch.optim.Adam(stage_parameters(self.model,"A"),lr=1e-4)}
        self.scaler=torch.amp.GradScaler(self.device.type,enabled=self.device.type=="cuda")
        self.logs=[];self.best_state=None;self.set_stage("A")

    def set_stage(self,stage):
        active={id(p) for p in stage_parameters(self.model,stage)}
        for p in self.model.parameters():p.requires_grad_(id(p) in active)
        self.model.zero_grad(set_to_none=True);self.model.train();self.stage=stage

    def transition(self,step,event):
        if step!=self.controller.schedule.A_steps+1 or event["stage"]!="B" or self.stage!="A" or self.controller.completed!=self.controller.schedule.A_steps:
            raise ValueError("invalid or repeated optimizer transition")
        self.set_stage("B")
        self.optimizers["B"]=torch.optim.Adam(stage_parameters(self.model,"B"),lr=1e-4)

    def _to_device(self,x):return x.to(self.device,non_blocking=self.device.type=="cuda")

    def matching_batch(self,branch,k):
        batch=self.matching.sample(branch,k)
        tensors=[self.sources.get(self.sources.allowed[d],"train").x[i] for d,i in batch["pairs"]]
        return (self._to_device(torch.stack(tensors)),torch.from_numpy(batch["y"]).to(self.device),
                torch.from_numpy(batch["d"]).to(self.device),torch.from_numpy(batch["anchor_mask"]).to(self.device))

    def main_batch(self):
        pairs=[]
        for _ in range(32):
            d=int(self.rng.randint(len(self.sources.allowed)))
            i=int(self.rng.randint(len(self.sources.get(self.sources.allowed[d],"train").y)))
            pairs.append((d,i));self.draws+=1
        x=torch.stack([self.sources.get(self.sources.allowed[d],"train").x[i] for d,i in pairs])
        y=torch.tensor([int(self.sources.get(self.sources.allowed[d],"train").y[i]) for d,i in pairs],device=self.device)
        d=torch.tensor([d for d,i in pairs],device=self.device)
        return self._to_device(x),y,d

    def _adversarial(self,x,labels,branch):
        encoder,head=(self.model.gf,self.model.hf) if branch=="f" else (self.model.gd,self.model.hd)
        direction=-1 if self.profile_name=="EQ_LITERAL" else 1
        with torch.autocast(self.device.type,enabled=False):
            query=x.detach().float().requires_grad_(True)
            gradient=torch.autograd.grad(F.cross_entropy(head(encoder(query)),labels),query)[0]
            perturbed=(x.detach().float()+direction*.1*gradient).detach()
            return F.mse_loss(encoder(perturbed),encoder(x.detach().float()))

    def optimizer_attempt(self,step,event):
        if self.permit is None:raise PermissionError("new finite authorization required")
        self.permit.require_neural(self.fingerprint)
        self.permit.claim(self.fingerprint,step,event["stage"])
        if event["stage"]!=self.stage:raise ValueError("optimizer stage and cursor disagree")
        self.model.zero_grad(set_to_none=True)
        stage=event["stage"]
        xf,yf,df,af=self.matching_batch("f",5 if stage=="A" else 2)
        xd,yd,dd,ad=self.matching_batch("d",5 if stage=="A" else 2)
        with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=="cuda"):
            mf=self.model.gf(xf);md=self.model.gd(xd)
            if stage=="A":mf=self.model.pf(mf);md=self.model.pd(md)
            lmf=matching_contrast(mf.float(),yf,df,"f",af)
            lmd=matching_contrast(md.float(),yd,dd,"d",ad)
            if stage=="A":terms={"lmf":lmf,"lmd":lmd};weights=None;total=lmf+lmd
            else:
                x,y,d=self.main_batch();gf=self.model.gf(x);gd=self.model.gd(x)
                terms={"lf":F.cross_entropy(self.model.hf(gf).float(),y),"ld":F.cross_entropy(self.model.hd(gd).float(),d),
                       "lmf":lmf,"lmd":lmd,"la":self._adversarial(x,y,"f")+self._adversarial(x,d,"d"),
                       "lcc":causal_consistency(gf.float(),gd.float(),y,d),"lrs":redundancy(gf.float(),gd.float()),
                       "lrc":F.mse_loss(self.model.dec(gf,gd).float(),x.float())}
                weights,total=weighted_total(list(terms.values()),detach_weights=self.profile_name!="EQ_LITERAL")
        if not torch.isfinite(total):raise RuntimeError("nonfinite loss")
        opt=self.optimizers[stage];self.scaler.scale(total).backward();self.scaler.unscale_(opt)
        norms={}
        for name in ("gf","gd","pf","pd","hf","hd","dec"):
            grads=[p.grad for p in getattr(self.model,name).parameters() if p.grad is not None]
            norm=torch.sqrt(sum(g.float().square().sum() for g in grads)) if grads else torch.tensor(0.)
            norms[name]=float(norm) if torch.isfinite(norm) else None
        first=stage_parameters(self.model,stage)[0];before=float(opt.state.get(first,{}).get("step",0))
        self.scaler.step(opt);self.scaler.update()
        actual=float(opt.state.get(first,{}).get("step",0))>before
        self.logs.append({"step":step,"stage":stage,"actual_update":actual,"amp_skip":not actual,
                          "losses":{k:float(v.detach()) for k,v in terms.items()},"total":float(total.detach()),
                          "weights":None if weights is None else weights.detach().cpu().tolist(),"gradient_norms":norms,
                          "BN_counts":{k:int(v) for k,v in self.model.state_dict().items() if k.endswith("num_batches_tracked")}})
        return actual

    @torch.no_grad()
    def refresh(self,step,event):
        if self.permit is None:raise PermissionError("authorized source refresh required")
        self.permit.require_neural(self.fingerprint)
        modes={n:m.training for n,m in self.model.named_modules()};self.model.eval()
        try:
            for branch in ("f","d"):
                arrays=[];encoder=getattr(self.model,"g"+branch);projector=getattr(self.model,"p"+branch)
                for domain in self.sources.allowed:
                    x=self.sources.get(domain,"train").x;chunks=[]
                    for offset in range(0,len(x),64):
                        self.permit.check(self.fingerprint)
                        z=encoder(self._to_device(x[offset:offset+64]))
                        if event["stage"]=="A":z=projector(z)
                        chunks.append(z.cpu().numpy())
                    arrays.append(np.concatenate(chunks))
                self.matching.refresh(arrays,branch,"projection" if event["stage"]=="A" else "encoder")
        finally:
            for n,m in self.model.named_modules():m.training=modes[n]

    @torch.no_grad()
    def source_probabilities(self,model=None):
        if self.permit is None:raise PermissionError("authorized source forward required")
        model=model or self.model;self.permit.require_neural(self.fingerprint)
        modes={n:m.training for n,m in model.named_modules()};model.eval();parts=[];labels=[]
        try:
            for domain in self.sources.allowed:
                windows=self.sources.get(domain,"val")
                for offset in range(0,len(windows.y),64):
                    self.permit.check(self.fingerprint)
                    x=self._to_device(windows.x[offset:offset+64])
                    parts.append(torch.softmax(model.hf(model.gf(x)),1).cpu().numpy())
                labels.append(windows.y)
        finally:
            for n,m in model.named_modules():m.training=modes[n]
        return np.concatenate(parts),np.concatenate(labels)

    def source_validate(self,step,event):
        probability,y=self.source_probabilities();score=float(np.mean(probability.argmax(1)==y))
        if self.controller.best_score is None or score>self.controller.best_score:
            self.best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        return {"source_score":score,"target_used_for_selection":False}

    def run(self,stop_after=None):
        if self.permit is None:raise PermissionError("no new training authorization")
        self.permit.require_neural(self.fingerprint)
        self.controller.run(self.optimizer_attempt,self.refresh,self.transition,self.source_validate,stop_after)

    def packet(self):
        cursor=self.controller.state_dict()
        event=self.controller.schedule.event(self.controller.completed) if self.controller.completed else {"outer":0,"inner":0}
        progress={"stage":self.stage,"outer":event["outer"],"inner":event["inner"],
                  "updates":self.controller.actual_updates,"cursor":cursor,"source_fingerprint":self.sources.fingerprint}
        selection={"best_step":self.controller.best_step if self.controller.best_step is not None else -1,
                   "best_score":self.controller.best_score if self.controller.best_score is not None else -1.,
                   "best_state":self.best_state,"target_used_for_selection":False}
        return capture_packet(self.model,self.optimizers,self.scaler.state_dict(),self.matching,
                              {"rng":self.rng.get_state(),"draws":self.draws,"logs":self.logs},
                              self.fingerprint,progress,selection,self.profile)

    def restore(self,packet):
        # All engine-level metadata is validated before the shared state restore.
        if packet["progress"]["source_fingerprint"]!=self.sources.fingerprint:
            raise ValueError("source tensor/split fingerprint mismatch")
        probe=ScheduleControl(self.fingerprint,self.controller.schedule)
        probe.load_state_dict(packet["progress"]["cursor"])
        if probe.failed_step is not None:raise ValueError("failed callback packet is diagnostic only")
        if probe.actual_updates!=packet["progress"]["updates"]:
            raise ValueError("optimizer/cursor count mismatch")
        expected_stage="A" if probe.completed<=probe.schedule.A_steps else "B"
        if packet["progress"]["stage"]!=expected_stage:
            raise ValueError("model/cursor stage mismatch")
        if probe.best_step!= (None if packet["selection"]["best_step"]==-1 else packet["selection"]["best_step"]):
            raise ValueError("source selection/cursor mismatch")
        expected_score=-1. if probe.best_score is None else probe.best_score
        if packet["selection"]["best_score"]!=expected_score:
            raise ValueError("source selection score/cursor mismatch")
        sampler=packet["sampler"]
        rng_probe=np.random.RandomState();rng_probe.set_state(sampler["rng"])
        if type(sampler["draws"]) is not int or sampler["draws"]<0 or not isinstance(sampler["logs"],list) or len(sampler["logs"])!=probe.completed:
            raise ValueError("sampler/progress logs inconsistent")
        if [r["step"] for r in sampler["logs"]]!=list(range(1,probe.completed+1)) or sum(int(r["actual_update"]) for r in sampler["logs"])!=probe.actual_updates:
            raise ValueError("attempt log/cursor counts inconsistent")
        expected_opts={"A"} if expected_stage=="A" else {"A","B"}
        if set(packet["optimizers"])!=expected_opts:raise ValueError("stage optimizer coverage")
        # Stage optimizer B on a local dict so bad packets do not mutate self.
        opts=dict(self.optimizers)
        if "B" in expected_opts and "B" not in opts:
            opts["B"]=torch.optim.Adam(stage_parameters(self.model,"B"),lr=1e-4)
        if set(opts)!=expected_opts:raise ValueError("restore into wrong active optimizer stage")
        restored=restore_packet(packet,self.model,opts,self.matching,self.fingerprint,self.profile,self.scaler)
        self.optimizers=opts;self.controller=probe;self.stage=expected_stage
        self.rng.set_state(restored["sampler"]["rng"]);self.draws=restored["sampler"]["draws"]
        self.logs=restored["sampler"]["logs"];self.best_state=restored["selection"]["best_state"]

    def save_resume(self,path,storage_root):
        target=Path(path).resolve();root=Path(storage_root).resolve()
        if root.drive.upper()!="D:" or not target.is_relative_to(root):
            raise PermissionError("new model state must stay inside named D storage")
        if target.exists():raise FileExistsError("resume output exists; do not overwrite")
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as stream:torch.save(self.packet(),stream)
