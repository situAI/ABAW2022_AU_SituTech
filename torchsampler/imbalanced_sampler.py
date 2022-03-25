class SamplerFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_by_name(dataset_name, dataset):
        if dataset_name == 'AffectNet_EXPR':
            from .imbalanced_SLML import ImbalancedDatasetSampler_SLML
            sampler = ImbalancedDatasetSampler_SLML(dataset)
        elif dataset_name == 'DISFA':#'Mixed_AU':
            from .imbalanced_ML import ImbalancedDatasetSampler_ML
            sampler = ImbalancedDatasetSampler_ML(dataset)
        elif dataset_name == 'AffectNet_VA':
            from .imbalanced_VA import ImbalancedDatasetSampler_VA
            sampler = ImbalancedDatasetSampler_VA(dataset)
        else:
            sampler = None
            #raise ValueError("Dataset [%s] not recognized." % dataset_name)
        return sampler
