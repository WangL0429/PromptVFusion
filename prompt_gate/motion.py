"""Frozen motion uses previous real observations and the synchronized pair."""
import torch

LANES=('visible','infrared')


class ObservedMotion:
    def __init__(self,estimator):
        self.estimator=estimator
        self.reset()

    def reset(self):
        self.next_index=0
        self.previous={}
        self.pair_images={}
        self.pair_index=None

    @torch.no_grad()
    def step(self,index,observed,mask):
        if index!=self.next_index:
            raise ValueError('Expected the next causal frame')
        images={lane:observed[lane] for lane,present in zip(LANES,mask) if present}
        if index==0 and not all(mask):
            raise ValueError('Initialize with both real modalities')
        reference=next(iter(images.values()))
        zero=reference.new_zeros(reference.shape[0],2,*reference.shape[-2:])
        common=[lane for lane in LANES if lane in images and lane in self.previous]
        # At a modality switch no consecutive real pair exists. Identity is a
        # deterministic fallback, never a claim of cross-modal correspondence.
        flow=(self.estimator(images[common[0]],self.previous[common[0]]).detach()
              if index and common else zero)
        if all(mask):
            pair_flow=zero
            self.pair_images={lane:image.detach().clone() for lane,image in images.items()}
            self.pair_index=index
        else:
            cue=next(iter(images))
            pair_flow=self.estimator(images[cue],self.pair_images[cue]).detach()
        self.previous={lane:image.detach() for lane,image in images.items()}
        self.next_index+=1
        return {'flow_prev':flow,'pair_flow':pair_flow}
