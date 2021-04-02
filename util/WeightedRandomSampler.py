import torch
from torch.utils.data import WeightedRandomSampler

class WeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, input_data, estimated_count_per_class):
        labels = input_data.labels

        weight_hate = 1 / labels.count(0)
        weight_offensive = 1 / labels.count(1)
        try:
            weight_neither = 1 / labels.count(2)
            weight = [weight_hate, weight_offensive, weight_neither]
        except:
            weight = [weight_hate, weight_offensive]

        samples_weight = torch.tensor([weight[t] for t in labels])
        super().__init__(weights=samples_weight, num_samples=estimated_count_per_class*3)

