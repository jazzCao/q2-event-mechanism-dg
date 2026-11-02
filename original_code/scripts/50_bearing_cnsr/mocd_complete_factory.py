"""Lazy, permit-gated seeded/instrumented wrapper over unchanged MOCD math.

The injectable class builder is for pure control fixtures. Public create()
imports neural dependencies only after the future runtime's actual permit.
No standalone training entry or human approval is supplied here.
"""
from contextlib import contextmanager
import random
import numpy as np
from mocd_complete_metadata import profile_streams


def engine_class(base,matching_class,torch):
    class SeededRecordedMOCD(base):
        def __init__(self,sources,profile_name,profiles,fingerprint,*,seed,permit,rt,on_source_probe,device,schedule):
            streams=profile_streams(seed)
            if permit is None or on_source_probe is None:raise PermissionError('explicit source permit and paid probe observer required')
            permit.require_neural(fingerprint)
            permit.validate_request(sources,profile_name,seed,profiles,schedule)
            if (len(sources.allowed),sources.classes) not in ((3,3),(2,8),(2,4)):raise ValueError('same three protocol source/class shapes')
            self.rt=rt;self.source_observer=on_source_probe;self.seed=seed;self.operation=None;self.last_probability=None
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            super().__init__(sources,profile_name,profiles,fingerprint,device=device,schedule=schedule,permit=permit)
            # Seed0 keeps the actual original matching/sampling objects intact.
            # For nonzero seeds, the constructor consumes matrix RNG draws and
            # the matching sampler continues from its saved post-construction RNG.
            if seed:
                self.matching=matching_class([sources.get(d,'train').y for d in sources.allowed],seed)
                self.rng=np.random.RandomState(seed)
            self.profile=dict(self.profile,complete_source_streams=streams)
            self._count_hooks=[getattr(self.model,name).register_forward_pre_hook(self._encoder_input) for name in ('gf','gd')]

        def _encoder_input(self,module,args):
            if self.operation not in ('training_encoder_windows','matching_encoder_windows','source_prediction_windows'):raise PermissionError('unclassified encoder forward')
            self.rt.claim(self.operation,len(args[0]))

        @contextmanager
        def counted(self,operation):
            previous=self.operation;self.operation=operation
            try:yield
            finally:self.operation=previous

        def optimizer_attempt(self,step,event):
            with self.counted('training_encoder_windows'):actual=super().optimizer_attempt(step,event)
            if type(actual) is not bool:raise ValueError('actual scaler/optimizer result required')
            # The optimizer action has already happened; future Runtime records
            # it before its post-action clock check, including AMP skips.
            self.rt.optimizer_finished(self.permit.name,step,actual)
            return actual

        def refresh(self,step,event):
            with self.counted('matching_encoder_windows'):value=super().refresh(step,event)
            self.permit.check(self.fingerprint)
            return value

        def source_probabilities(self,model=None):
            with self.counted('source_prediction_windows'):prob,y=super().source_probabilities(model)
            self.last_probability=prob,y
            return prob,y

        def source_validate(self,step,event):
            self.rt.source_probe_begin(self.permit.name,step)
            row=super().source_validate(step,event);prob,y=self.last_probability
            groups=np.concatenate([self.sources.get(d,'val').groups for d in self.sources.allowed])
            receipt=self.source_observer(step,prob,y,groups,row)
            self.rt.source_probe_finished(self.permit.name,step,receipt)
            return row

    return SeededRecordedMOCD


def create(sources,profile_name,profiles,fingerprint,*,seed,permit,rt,on_source_probe,schedule,device='cuda'):
    profile_streams(seed)
    if device!='cuda':raise PermissionError('no silent CPU neural fallback')
    if permit is None:raise PermissionError('no implicit finite neural permission')
    permit.require_neural(fingerprint)
    permit.validate_request(sources,profile_name,seed,profiles,schedule)
    rt.gpu();rt.claim('neural_model_builds')
    # These imports are deliberately unreachable in preparation/pure fixtures.
    import torch
    from mocd_full_source_engine import RecordedFullSourceEngine
    from mocd_full_matching import BlockedIndexedMatching
    cls=engine_class(RecordedFullSourceEngine,BlockedIndexedMatching,torch)
    return cls(sources,profile_name,profiles,fingerprint,seed=seed,permit=permit,rt=rt,on_source_probe=on_source_probe,device=device,schedule=schedule)
