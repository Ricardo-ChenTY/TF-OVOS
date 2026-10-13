#!/usr/bin/env python3
"""Restore only the omitted official VOC slide_crop=0; retain benchmark names."""
import sys
import measure_e4 as engine
import measure_e4_full as worker

BaselineDense=engine.Dense
class ReleaseVOC(BaselineDense):
    def __init__(self,method,ds,*args,**kwargs):
        assert method=='resclip' and ds=='voc20'
        super().__init__(method,ds,*args,**kwargs)
        assert self.model.slide_crop==224
        baseline='--baseline-replay' in sys.argv
        if not baseline:self.model.slide_crop=0
        self.meta['model']['slide_crop']=self.model.slide_crop
        self.meta['correction']=dict(parameter='slide_crop',historical=224,official=0,
                                     applied=not baseline,scope='VOC only; benchmark vocabulary unchanged')
engine.Dense=ReleaseVOC
if __name__=='__main__':
    baseline='--baseline-replay' in sys.argv
    if baseline:sys.argv.remove('--baseline-replay')
    # Preserve the flag for the backend without exposing it to the common parser.
    original=ReleaseVOC.__init__
    def init(self,*args,**kwargs):
        if baseline:sys.argv.append('--baseline-replay')
        try:original(self,*args,**kwargs)
        finally:
            if baseline:sys.argv.remove('--baseline-replay')
    ReleaseVOC.__init__=init
    worker.main()
