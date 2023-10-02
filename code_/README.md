## 团队名称与队长联系方式

团队名称：BTBU2Causality

队长联系方式：
	姓名：yana
	微信：wxid_0qromn1o2wdr21
	手机：13580479946

## 镜像说明

镜像有修改（抱歉，主要是旧版本pytorch没有的功能，只有nightly版本有）:

python==3.8.0
安装命令 `conda create -n <env_name> python=3.8.0`

torch==2.2.0, torchaudio==2.2.0, torchvision==0.17.0
安装命令 `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118`

yacs, termcolor
安装命令 `pip install -r requirements.txt`

## 预计的训练时间

4splitdomain——40分钟
10splittask——4小时

## 其他细节

 `code_/agents/`下添加 `worker.py`文件，继承了 `trainer.py`中的一些方法

`utils/`下的参数文件 `user_4splitDomains.yaml`和 `user_10splitTasks.yaml`有修改

`utils/`下的参数文件 `config.py`有修改：调用worker和CLworker，使用了权重衰减
