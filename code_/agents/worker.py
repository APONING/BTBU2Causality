# prototype策略: 保存所有特征 
import torch
from .trainer import CLTrainer
import os
import torch
import torch.distributed as dist
import copy
from datasets.base import get_this_dataset
from datasets.transforms import val_transforms_l
from torch.utils.data import DataLoader,Dataset


class CLWorker(CLTrainer):
    """_summary_

    Args:
        CLTrainer (_type_): _description_
    """
    def __init__(self, config, args, logger, out_dim, ckpt_path=None):
        super().__init__(config, args, logger, out_dim, ckpt_path)
        self.config_hat = config
        # os.makedirs(os.path.dirname(args.save_ckpt_path), exist_ok=True)
        
    def init_optimizer(self):
        if self.task_count == 0:
            optimizer_arg = {'params': self.model.parameters(),
                             'lr': self.config['lr'],
                             'weight_decay': self.config['weight_decay']}
        else:
            optimizer_arg = {'params': self.model.last.parameters(),
                             'lr': self.config['lr'],
                             'weight_decay': self.config['weight_decay']}            
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=self.config['gamma'])
    
    
    def before_tasks(self, train_loader_hat):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            # if self.args.is_main_process:
            self.log.info('Optimizer is reset!')
            self.init_optimizer()  # 2/2 第二次使用
        feature_loader = None
        if self.task_count > 0 and not self.multihead:
            for name, param in self.model.named_parameters():
                if "last" not in name:
                    param.requires_grad = False
            if not self.multihead:  # domain incremental
                # 构造训练特征集 保存所有历史特征
                orig_mode = self.model.training
                # 构造新任务特征 1/2：提取样本特征 
                self.model.eval()
                _vectors, _targets, _task = [], [], []
                for _input, _target, _task in train_loader_hat:
                    with torch.no_grad():
                        _vec = self.model.forward_features(_input.to(self.device))
                    _vectors.append(_vec.cpu())
                    _targets.append(_target)
                _vectors = torch.cat(_vectors)
                _targets = torch.cat(_targets)
                self._vectors = torch.cat([self._vectors, _vectors], dim=0)
                self._targets = torch.cat([self._targets, _targets], dim=0)
                # 构造特征集
                feature_set =  FeatureDataset(self._vectors, self._targets)
                feature_loader = DataLoader(feature_set, batch_size=self.config_hat.DATASET.BATCHSIZE, shuffle=True, num_workers=self.config_hat.DATASET.NUM_WORKERS, pin_memory=True)
                self.model.train(orig_mode)
        return feature_loader
           
    def learn_tasks(self, train_loader, val_loader=None):
        if not self.multihead:  # domian incremental
            tab_name_ = 'CL4SplitDomains'
            train_dataset = get_this_dataset('train', str(self.task_count), self.config_hat.DATASET.ROOT,
                                             tab_name_, val_transforms_l())
            train_loader_hat =  torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=self.config_hat.DATASET.BATCHSIZE,
                                                    shuffle=False,
                                                    sampler = None,
                                                    num_workers=self.config_hat.DATASET.NUM_WORKERS,
                                                    pin_memory = True)
        else:
            train_loader_hat = None
        feature_loader = self.before_tasks(train_loader_hat)
        self.train_model(train_loader, val_loader, train_loader_hat, feature_loader)
        self.after_tasks(train_loader)

    def train_model(self, train_loader, val_loader, train_loader_hat=None, feature_loader=None):
        acc_best = 0
        for epoch in range(self.config['schedule'][-1]):
            # Config the model and optimizer
            # if self.args.is_main_process:
            self.log.info('Epoch:{0}'.format(epoch))
            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch)
            self.model.train() if self.task_count==0 else self.model.eval()
            # self.scheduler.step(epoch)  # 报错 放到 train_epoch() 后
            # if self.args.is_main_process:
            for param_group in self.optimizer.param_groups:
                self.log.info('LR:{}'.format(param_group['lr']))

            if self.args.is_main_process:
                log_str = ' Itr\t    Time  \t    Data  \t  Loss  \t  Acc'
            if self.args.distributed:
                log_str = 'Rank\t' + log_str
            self.log.info(log_str)

            self.before_epoch()
            self.train_epoch(feature_loader) if not self.multihead and self.task_count > 0 else self.train_epoch(train_loader)
            self.scheduler.step()

            if self.args.distributed:
                dist.barrier()

            self.after_epoch()
            # Evaluate the performance of current task
            if val_loader != None:
                # self.validation(val_loader)
                acc_e = self.validation(val_loader)
                if not acc_best > acc_e:
                      model_best = copy.deepcopy(self.model)
                      acc_best = acc_e 
                      epoch_best = epoch                     
            else:
                model_best = copy.deepcopy(self.model)
                epoch_best = None
            self.log.info("Epoch_best-{}: acc_best={:.2f} \t Epoch-{}: acc_e={:.2f}".format(epoch_best, acc_best, epoch, acc_e))
        self.model = model_best       
                
        if not self.multihead and self.task_count==0:
            self._compute_vectors(train_loader_hat)  # 初始化任务特征
    
    
    def update_model(self, inputs, targets, tasks):
        if self.task_count > 0 and not self.multihead:
            out = self.model.logits(inputs)
        else:
            out = self.forward(inputs)            
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out    
       
    def _compute_vectors(self, train_loader_hat):  # 保存所有历史特征
        # 得到所有样本的特征 vectors 和标签 targets
        orig_mode = self.model.training
        self.model.eval()
        vectors, targets = [], []
        for _input, _target, _ in train_loader_hat:
            with torch.no_grad():
                _vec = self.model.forward_features(_input.to(self.device))
            vectors.append(_vec.cpu())
            targets.append(_target)
        self._vectors = torch.cat(vectors)
        self._targets = torch.cat(targets)
        self.model.train(orig_mode)
  
    def collect_memoey(self):
        """
            Return a variable of dictionary type,
            each value in the dictionary should be of type ``float32".
        """
        storage = None if self.multihead else {'vectors': self._vectors, 'targets': self._targets}
        return storage
    
    def allocate_memory(self, storage):
        if not self.multihead:            
            self._vectors = storage['vectors']
            self._targets = storage['targets']
        
        
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label, idx