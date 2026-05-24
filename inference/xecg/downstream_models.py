from xECG import xECG
import torch
import torch.nn as nn

class xECGClassification(xECG):
    def __init__(
            self, 
            config,
            num_classes,
            linear_probing=False,
            cls_type='avg',
        ): 
        self.linear_probing = linear_probing
        super(xECGClassification, self).__init__(cls_type=cls_type, config=config)

        self.head = nn.Sequential(
            get_normalization_layer(config, config['embedding_size']),
            nn.Linear(config['embedding_size'], num_classes)
        )

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                cls, _ = super().forward(x)
        else:  
            cls, _ = super().forward(x)

        res = self.head(cls)
        return res

class xECGFeatureClassification(xECG):
    def __init__(
            self, 
            config,
            num_classes,
            linear_probing=False,
        ): 
        self.linear_probing = linear_probing
        super(xECGFeatureClassification, self).__init__(cls_type=None, config=config)

        self.head = nn.Sequential(
            get_normalization_layer(config, config['embedding_size']),
            nn.Linear(config['embedding_size'], num_classes)
        )

    def forward(self, x):
        if self.linear_probing:
            with torch.no_grad():
                _, features = super().forward(x)
        else:  
            _, features = super().forward(x)


        res = self.head(features)
        return res
    


def get_normalization_layer(config, embedding_size=None):
    if config['cls_normalization'] == 'layer':
        return nn.LayerNorm(embedding_size)
    elif config['cls_normalization'] == 'batch':
        return nn.BatchNorm1d(embedding_size)
    elif config['cls_normalization'] == 'instance':
        return nn.InstanceNorm1d(embedding_size)
    else:
        return nn.Identity()