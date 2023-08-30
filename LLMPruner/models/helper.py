import torch

def reorder_qkv(qkv_module, head_dim, num_head):
    """
    Reorder the index of query-key-value.
    The reason why we need this step: The tracing of torch.view is too complicated to trace, and thus, the index mapping of query_key_value would have some problem.
    The Goal of this function is to reorder the concatenated query-key-value 
    from
    [q1, k1, v1, q2, k2, v2, ...] 
    to 
    [q1, q2, ..., k1, k2, ..., v1, v2, ...]
    """
    #batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    #fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim) # LLM-Pruner: Modify the split of query_key_value here
    #return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :] 

    weight = qkv_module.weight
    bias = qkv_module.bias
    out_features, in_features = weight.shape
    assert bias.shape == (out_features, )

    head_dim_qkv_head = weight.view(num_head, -1, head_dim, in_features)
    q, k, v = head_dim_qkv_head[..., 0, :, :], head_dim_qkv_head[..., 1, :, :], head_dim_qkv_head[..., 2, :, :]
    new_weight = torch.cat([q, k, v], dim=0).view(out_features, in_features)
    qkv_module.weight = torch.nn.Parameter(new_weight) # out X in

    bias = bias.view(num_head, -1, head_dim)
    q_bias, k_bias, v_bias = bias[..., 0, :], bias[..., 1, :], bias[..., 2, :]
    new_bias = torch.cat([q_bias, k_bias, v_bias], dim=0).view(out_features)
    qkv_module.bias = torch.nn.Parameter(new_bias) # out







