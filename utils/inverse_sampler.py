# Code adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/posterior_sample.py
# Original author: bingliang

from .logging import Trajectory
import torch


def sample_in_batch(sampler,
                    model,
                    x_start,
                    operator,
                    y,
                    evaluator,
                    verbose,
                    record,
                    batch_size,
                    gt,
                    **kwargs):
    """
        posterior sampling in batch
    """
    samples = []
    trajs = []
    for s in range(0, len(x_start), batch_size):
        # update evaluator to correct batch index
        cur_x_start = x_start[s:s + batch_size]
        if type(y) in [tuple, list]:
            cur_y = tuple([y_part[s:s + batch_size] for y_part in y])
        else:
            cur_y = y[s:s+batch_size]
        
        cur_gt = gt[s: s + batch_size]
        cur_samples = sampler.sample(model, cur_x_start, operator, cur_y,
                                     evaluator, verbose=verbose, record=record,
                                     gt=cur_gt, **kwargs)

        samples.append(cur_samples)
        if record:
            trajs.append(sampler.trajectory.compile())
    if record:
        trajs = Trajectory.merge(trajs)
    return torch.cat(samples, dim=0), trajs
