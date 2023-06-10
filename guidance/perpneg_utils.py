import torch

# Please refer to the https://perp-neg.github.io/ for details about the paper and algorithm
def get_perpendicular_component(x, y):
    assert x.shape == y.shape
    return x - ((torch.mul(x, y).sum())/max(torch.norm(y)**2, 1e-6)) * y


def batch_get_perpendicular_component(x, y):
    assert x.shape == y.shape
    result = []
    for i in range(x.shape[0]):
        result.append(get_perpendicular_component(x[i], y[i]))
    return torch.stack(result)


def weighted_perpendicular_aggregator(delta_noise_preds, weights, batch_size):
    """ 
    Notes: 
     - weights: an array with the weights for combining the noise predictions
     - delta_noise_preds: [B x K, 4, 64, 64], K = max_prompts_per_dir
    """
    delta_noise_preds = delta_noise_preds.split(batch_size, dim=0) # K x [B, 4, 64, 64]
    weights = weights.split(batch_size, dim=0) # K x [B]
    # print(f"{weights[0].shape = } {weights = }")

    assert torch.all(weights[0] == 1.0)

    main_positive = delta_noise_preds[0] # [B, 4, 64, 64]

    accumulated_output = torch.zeros_like(main_positive)
    for i, complementary_noise_pred in enumerate(delta_noise_preds[1:], start=1):
        # print(f"\n{i = }, {weights[i] = }, {weights[i].shape = }\n")

        idx_non_zero = torch.abs(weights[i]) > 1e-4
        
        # print(f"{idx_non_zero.shape = }, {idx_non_zero = }")
        # print(f"{weights[i][idx_non_zero].shape = }, {weights[i][idx_non_zero] = }")
        # print(f"{complementary_noise_pred.shape = }, {complementary_noise_pred[idx_non_zero].shape = }")
        # print(f"{main_positive.shape = }, {main_positive[idx_non_zero].shape = }")
        if sum(idx_non_zero) == 0:
            continue
        accumulated_output[idx_non_zero] += weights[i][idx_non_zero].reshape(-1, 1, 1, 1) * batch_get_perpendicular_component(complementary_noise_pred[idx_non_zero], main_positive[idx_non_zero])
    
    assert accumulated_output.shape == main_positive.shape, f"{accumulated_output.shape = }, {main_positive.shape = }"


    return accumulated_output + main_positive