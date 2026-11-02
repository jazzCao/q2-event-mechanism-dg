"""One fixed exposure policy and its expected-energy control. No I/O or ML."""
import math
import numpy as np
from source_rate_core import shared_edit_mask

ARMS=('T0_HALF_EXPOSURE','M0_EXPECTED_ENERGY_SHRINK')

def replacement(arm,clean,edited,uniforms,domains,classes,per_cell,*,to_mask=lambda v:v,where=np.where):
    if clean.shape!=edited.shape or clean.ndim!=2:
        raise ValueError('same bank clean and current Q2 vectors')
    if len(clean)!=domains*classes*per_cell:
        raise ValueError('complete balanced source batch')
    if arm==ARMS[0]:
        keep=shared_edit_mask(uniforms,domains,classes,per_cell)
        return where(to_mask(keep)[:,None],edited,clean),keep
    if arm==ARMS[1]:
        return clean+math.sqrt(.5)*(edited-clean),np.ones(len(clean),dtype=bool)
    if arm=='Q2':return edited,np.ones(len(clean),dtype=bool)
    raise ValueError('only fixed exposure, shrink control or unchanged reference')
