import sys, os
import numpy as np
import torch
import torch.nn as nn
from . import bit_models
from .Resnet import resnet152, resnet50
from torchvision.models import mobilenet_v2

    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# IMAGENET PERFORMANCE FOR TIMM MODELS IS HERE:
# https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
def get_arch(model_name, in_c=3, n_classes=1, do_multigranularities=False, n_step=4, do_multilabel=False, number_multi_label=[2, 7, 13, 23]):

    num_ftr = [512, 1024, 2048]
    hidden_feature_size = 512
    reduced_out_feature_size = 1024
    # number_multi_label = [2, 7, 13, 23]
    
    
    def build_reduce_pooling(num_ftr):
        reduce_pooling = []
        for ftr_in in num_ftr:
            reduce_pooling.append(nn.Sequential(
                BasicConv(ftr_in, hidden_feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(hidden_feature_size, reduced_out_feature_size, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveAvgPool2d(output_size=1)
            ))
        
        return reduce_pooling
        
    def build_classifier_feature(reduced_out_feature_size):
        
        return nn.Sequential(
            nn.BatchNorm1d(reduced_out_feature_size),
            nn.Linear(reduced_out_feature_size, hidden_feature_size),
            nn.BatchNorm1d(hidden_feature_size),
            nn.ELU(inplace=True),
        )
        
    def build_classifier(n_class, k=1, n=1):
        
        return nn.Linear(hidden_feature_size // n * k, n_class)
    
    def root2leaf(x):
        l1 = x[:, 0:hidden_feature_size // 4]
        l2 = x[:, hidden_feature_size // 4:hidden_feature_size // 4 * 2]
        l3 = x[:, hidden_feature_size // 4 * 2:hidden_feature_size // 4 * 3]
        l4 = x[:, hidden_feature_size // 4 * 3:hidden_feature_size]
                
        l1 = torch.cat([l1, l2.detach(), l3.detach(), l4.detach()], 1)
        l2 = torch.cat([l2, l3.detach(), l4.detach()], 1)
        l3 = torch.cat([l3, l4.detach()], 1)
        return [l1, l2, l3, l4]
    
    class M3(nn.Module):
        def __init__(self, num_ftr = [512, 1024, 2048],
                            hidden_feature_size = 512,
                            reduced_out_feature_size = 1024,
                            number_multi_label = [2, 7, 13, 23],
                            model_name = model_name,
                            do_multigranularities = do_multigranularities,
                            n_step = 4,
                            do_multilabel = do_multilabel):
            super().__init__()
                
            if model_name == 'mobilenetV2':
                self.model = mobilenet_v2(pretrained=True)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier = torch.nn.Linear(num_ftrs, n_classes)
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

            elif model_name == 'bit_resnext50_1':
                bit_variant = 'BiT-M-R50x1'
                self.model = bit_models.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
                if not os.path.isfile('models/BiT-M-R50x1.npz'):
                    print('downloading bit_resnext50_1 weights:')
                    os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz -P models/')
                self.model.load_from(np.load('models/BiT-M-R50x1.npz'))
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                                
                def build_feature_extracter(model):
                    self.f0 = lambda x: model.body.block1(model.root(x))
                    self.f1 = lambda x: model.body.block2(x)
                    self.f2 = lambda x: model.body.block3(x)
                    self.f3 = lambda x: model.body.block4(x)
                
                delattr(self.model, "head")
                
            elif model_name == 'resnet152':
                self.model = resnet152(pretrained=True)
                self.model.fc = torch.nn.Linear(2048, 23)
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                                
                def build_feature_extracter(model):                        
                    self.f0 = lambda x: model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
                    self.f1 = lambda x: model.layer2(x)
                    self.f2 = lambda x: model.layer3(x)
                    self.f3 = lambda x: model.layer4(x)    
                    
            elif model_name == 'resnet50':
                self.model = resnet50(pretrained=True)
                self.model.fc = torch.nn.Linear(2048, 23)
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                                
                def build_feature_extracter(model):                        
                    self.f0 = lambda x: model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
                    self.f1 = lambda x: model.layer2(x)
                    self.f2 = lambda x: model.layer3(x)
                    self.f3 = lambda x: model.layer4(x)
            
            else:
                sys.exit('not a valid model_name, check models.get_model.py')
            
            self.n_classes = n_classes
            self.number_multi_label = number_multi_label
            self.mean = mean
            self.std = std
            self.do_multigranularities = do_multigranularities
            self.n_step = n_step
            self.do_multilabel = do_multilabel
            
            if self.do_multigranularities is True:
                ## extract feature -> reduce and pooling -> flatten -> expand
                # feature_extract = extract_last_three_stage_feature
                assert self.n_step >= 2 and self.n_step <= 4
                assert self.n_step <= len(num_ftr) + 1
                reduce_pooling = build_reduce_pooling(num_ftr[-(self.n_step-1):]) # [rp_1, rp_2, rp_(n_step-1)]
                rp_out = [[1, 1], [1, 1, 2], [1, 1, 1, 3]]
                classifier_root = [build_classifier_feature(reduced_out_feature_size * k) for k in rp_out[n_step-2]] # [f1, f2, f3, f_concate]
                self.n_root = self.n_step
                
            else:
                # feature_extract = extract_last_stage_feature
                reduce_pooling = build_reduce_pooling([num_ftr[-1]]) # [rp1]
                classifier_root = [build_classifier_feature(reduced_out_feature_size)] # [f1]
                self.n_root = 1
            
            if self.do_multilabel is True:
                classifiers = [[build_classifier(n_class, len(number_multi_label) - i, len(number_multi_label)) for i, n_class in enumerate(number_multi_label)] for _ in range(self.n_root)]   # 4 * 4 at most, for example
                                                                # [[c_s1_l1, c_s1_l2, ...], # leaves 1, 2 with root 1
                                                                #  [c_s2_l1, c_s2_l2, ...],
                                                                #            ...          ,
                                                                #            ...           ]
                self.n_label = 4
            else:
                classifiers = [[build_classifier(number_multi_label[-1])] for _ in range(self.n_root)] # 4 at most
                self.n_label = 1
            
            
            ## set functions and nets
            build_feature_extracter(self.model)
            
            self.params_to_learn = []
            
            for i, rp in enumerate(reduce_pooling):
                setattr(self, "reduce_pooling_s{}".format(1 + i), rp)
                self.params_to_learn.append("reduce_pooling_s{}".format(1 + i))
            for i, bc in enumerate(classifier_root):
                setattr(self, "classifier_root_s{}".format(1 + i), bc)
                self.params_to_learn.append("classifier_root_s{}".format(1 + i))
            for i, c in enumerate(classifiers):
                for j, c_ in enumerate(c):
                    setattr(self, "classifier_leaf_s{}_l{}".format(1 + i, 1 + j), c_)
                    self.params_to_learn.append("classifier_leaf_s{}_l{}".format(1 + i, 1 + j))
            
            # params summary
            for key, value in self.named_parameters(): # debug parameter names
                print(key, value.shape)
            
            # params to learn summary
            for key in self.params_to_learn:
                print(key)
            
                
        def forward(self, x, stage=4):
            ''' Only training selected stage:
            Stage within [1,2,3,4,5]. Training stage within [1,2,3,4]. Inference stage is 5.
            If stage < 4, only forward selected stage. For example, if stage is 1, skip stage [1, 2, 4]. y=1x4
            if stage == 4, don't skip reduce_pooling_s of stage [1,2,3], but skip classifier_root_s and classifier_leaf_s_l of stage [1,2,3]. y=1x4
            if stage == 5, forward all the nets. y=4x4
            '''
            if self.do_multigranularities is True: # PMG with or without FGN
                x = self.extract_last_three_stage_feature(x)
                x_concat = []
                y = [] # [[y_s1_l1, y_s1_l2, ...], [y_s2_l1, ...], ...]
                
                for i, x_reduce_pooling_feature in enumerate(x[-(self.n_step-1):]): # length of x is (self.n_step-1)
                    if stage < self.n_step and 1 + i != stage:
                        continue
                    
                    else:
                        x_reduce_pooling_feature = getattr(self, "reduce_pooling_s{}".format(1 + i))(x_reduce_pooling_feature)
                        x_reduce_pooling_feature = x_reduce_pooling_feature.view(x_reduce_pooling_feature.size(0), -1)
                        x_concat.append(x_reduce_pooling_feature)
                        
                        if stage == self.n_step:
                            continue
                        
                        else:
                            x_classifier_root = getattr(self, "classifier_root_s{}".format(1 + i))(x_reduce_pooling_feature)
                            x_leaf = root2leaf(x_classifier_root) if self.do_multilabel else [x_classifier_root]
                            y.append([getattr(self, "classifier_leaf_s{}_l{}".format(1 + i, 1 + j))(x_leaf[j]) for j in range(self.n_label)])
                
                if stage >= self.n_step:
                    x_concat = torch.cat(x_concat, -1)
                    x_classifier_root = getattr(self, "classifier_root_s{}".format(self.n_step))(x_concat)
                    x_leaf = root2leaf(x_classifier_root) if self.do_multilabel else [x_classifier_root]
                    y.append([getattr(self, "classifier_leaf_s{}_l{}".format(self.n_step, 1 + j))(x_leaf[j]) for j in range(self.n_label)])
                
                return y
                                
            else: # without PMG, with or without FGN
            
                x = self.extract_last_stage_feature(x)[-1]
                x = getattr(self, "reduce_pooling_s{}".format(1))(x)
                x = x.view(x.size(0), -1)
                
                x = getattr(self, "classifier_root_s{}".format(1))(x)
                x_leaf = root2leaf(x) if self.do_multilabel else [x]
                y = [[getattr(self, "classifier_leaf_s{}_l{}".format(1, 1 + j))(x_leaf[j]) for j in range(self.n_label)]]

                return y
        
        
        ## init functions and nets
        def extract_last_stage_feature(self, x):
            x = self.f0(x)
            x = self.f1(x)
            x = self.f2(x)
            x = self.f3(x)
                        
            return [x]
            
        def extract_last_three_stage_feature(self, x):
            x0 = self.f0(x)
            x1 = self.f1(x0)
            x2 = self.f2(x1)
            x3 = self.f3(x2)
                        
            return [x1, x2, x3]
        
    model = M3(model_name=model_name, do_multigranularities=do_multigranularities, n_step=n_step, do_multilabel=do_multilabel,
               number_multi_label=number_multi_label)
    mean = model.mean
    std = model.std
    
    return model, mean, std


