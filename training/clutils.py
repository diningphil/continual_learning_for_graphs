import torch
import os

def zerolike_params_dict(model):
    return [ ( k, torch.zeros_like(p).to(p.device) ) for k,p in model.named_parameters() ]


def copy_params_dict(model, copy_grad=False):
    if copy_grad:
        return [ ( k, p.grad.data.clone() ) for k,p in model.named_parameters() ]
    else:
        return [ ( k, p.data.clone() ) for k,p in model.named_parameters() ]

def normalize_blocks(importance):
    """
    0-1 normalization over each parameter block
    :param importance: [ (k1, p1), ..., (kn, pn)] (key, parameter) list
    """

    for _, imp in importance:
        max_imp, min_imp = imp.max(), imp.min()
        imp -= min_imp
        imp /= float(max(max_imp - min_imp, 1e-6))
        
    return importance


def save_importance(writer, importance, task_id):
    for paramname, imp in importance:
        if len(imp.size()) == 1: # bias
            writer.add_image(f"{paramname}_importance/{task_id}", imp.unsqueeze(0).cpu().data, 0, dataformats='HW')
        else:
            writer.add_image(f"{paramname}_importance/{task_id}", imp.cpu().data, 0, dataformats='HW')
        writer.add_histogram(f"{paramname}_importance_hist/{task_id}", imp.cpu().view(-1).data, 0)
            


