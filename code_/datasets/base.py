import os
from io import BytesIO
import os.path as osp
from math import ceil
from random import shuffle
import sqlite3
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import train_transforms_l,  val_transforms_l, test_transforms_l


# https://blog.csdn.net/shiwanghualuo/article/details/120778553
class SQLDataset(Dataset):
    def __init__(self, db_path, table_name, task_name, train_val=True, transform=None):
        super().__init__()
        self.db_path = db_path
        self.conn = None 
        self.establish_conn()
        self.train_val = train_val
        
        self.table_name = table_name
        self.cursor.execute(f'select max(rowid) from {self.table_name}')
        self.nums = self.cursor.fetchall()[0][0]
        
        self.transform = transform
        self.task_name = task_name
    
    def __getitem__(self, index):
        self.establish_conn()
        
        # inquery
        search_sql = f'select * from {self.table_name} where rowid=?'
        self.cursor.execute(search_sql, (index+1,))
        
        if self.train_val:
            img_bytes, label, task_name = self.cursor.fetchone()
        else:
            img_bytes, task_name = self.cursor.fetchone()
        
        assert str(task_name) == self.task_name
        
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        task_name = str(task_name)
        if self.train_val:
            return img, label, task_name
        else:
            return img, task_name
    
    def __len__(self):
        return self.nums
    
    def establish_conn(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path,
                                        check_same_thread=False,
                                        cached_statements=1024)
            self.cursor = self.conn.cursor()
        return self

    def close_conn(self):
        if self.conn is not None:
            self.cursor.close()
            self.conn.close()
            del self.conn 
            self.conn = None 
        return self
 
def get_this_dataset(stage, task_name, root, table_name_, trans, train_val=True):
    task_path = osp.join(root, f'{stage}_{task_name}.db')
    table_name = table_name_ + f'{stage}_{task_name}'
    dataset = SQLDataset(task_path, table_name=table_name, task_name=task_name, transform=trans, train_val=train_val)
    dataset.close_conn()
    return dataset


def get_10splitTasks(root, stage, task_name):
    tab_name_ = 'CL10SplitTasks'

    if stage=='train':
        return get_this_dataset(stage, task_name, root, tab_name_, train_transforms_l())
    elif stage=='val':
        return get_this_dataset(stage, task_name, root, tab_name_, val_transforms_l())
    elif stage == 'test':
        return get_this_dataset(stage, task_name, root, tab_name_, test_transforms_l(), train_val=False)
    else:
        raise 'Invalid stage!'

    return get_this_dataset(stage, task_name, root, tab_name_, trans)
    
def get_4splitDomains(root, stage, task_name):
    tab_name_ = 'CL4SplitDomains'

    if stage=='train':
        return get_this_dataset(stage, task_name, root, tab_name_, train_transforms_l())
    elif stage=='val':
        return get_this_dataset(stage, task_name, root, tab_name_, val_transforms_l())
    elif stage == 'test':
        return get_this_dataset(stage, task_name, root, tab_name_, test_transforms_l(), train_val=False)
    else:
        raise 'Invalid stage!'

    return get_this_dataset(stage, task_name, root, tab_name_, trans)
  