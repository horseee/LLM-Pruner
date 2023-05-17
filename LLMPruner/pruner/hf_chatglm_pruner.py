import torch
import torch.nn as nn

import LLMPruner.torch_pruning as tp
from LLMPruner.torch_pruning import BasePruningFunc, ops

from copy import deepcopy
import random
from functools import reduce
from operator import mul

from typing import Callable, Sequence, Tuple, Dict
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

##############################
# Pruners
##############################

##############################
# Importance
##############################
class MagnitudeImportance(tp.importance.Importance):
    def __init__(self, p=2, group_reduction="mean", normalizer=None):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
    
        group_imp = []
        #Get group norm
        #print("Group: ", group)
        #exit()
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            # Linear out_channels
            if prune_fn in [tp.prune_linear_out_channels,]:
                w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                group_imp.append(local_norm)
            # Linear in_channels
            elif prune_fn in [tp.prune_linear_in_channels,]:    
                print(dep, layer.weight.shape, len(idxs))
                w = layer.weight
                local_norm = w.abs().pow(self.p).sum(0)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)
            # Embedding
            elif prune_fn == tp.prune_embedding_out_channels:
                w = layer.weight.data[:, idxs]
                local_norm = w.abs().pow(self.p)
                group_imp.append(local_norm)

        if len(group_imp)==0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp

class TaylorImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum", normalizer=None, taylor=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.taylor = taylor

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction=='second':
            group_imp = group_imp[1]
        elif self.group_reduction=='customized':
            if group_imp.size(0) == 3:
                group_imp = group_imp[2]
            elif group_imp.size(0) == 4:
                group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
    
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

           
            if prune_fn not in [
                tp.prune_linear_out_channels, tp.prune_linear_in_channels, 
            ]:
                continue

            w = layer.weight
            grad = layer.weight.grad
            salience = w * grad

            if prune_fn in [tp.prune_linear_out_channels, ]:
                #second_order = torch.diagonal(w @ (grad.T @ grad) @ w.T)
                #local_norm = salience.abs().sum(1) + 0.5 * second_order
                if self.taylor == 'first-order':
                    local_norm = salience.sum(1).abs()
                elif self.taylor == 'hessian':
                    local_norm = salience.pow(2).sum(1)
                elif self.taylor == 'aprox-fisher':
                    local_norm = salience.abs().sum(1) 
                else:
                    raise NotImplementedError
                #print(local_norm, local_norm.shape)
                group_imp.append(local_norm)
            # Linear in_channels
            elif prune_fn in [tp.prune_linear_in_channels, ]:
                #second_order = torch.diagonal(w.T @ (grad @ grad.T) @ w)   
                #local_norm = salience.abs().sum(0)  + 0.5 * second_order
                if self.taylor == 'first-order':
                    local_norm = salience.sum(0).abs()
                elif self.taylor == 'hessian':
                    local_norm = salience.pow(2).sum(0)
                elif self.taylor == 'aprox-fisher':
                    local_norm = salience.abs().sum(0)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        if len(group_imp)==0:
            return None

        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp