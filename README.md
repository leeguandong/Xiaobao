[📘使用文档]() |
[🛠安装教程]() |
[👀模型库]() |
[🆕更新日志]() |
[🚀进行中的项目]() |
[🤔报告问题]()

</div>

 ## 简介
古代民间自媒体叫小报，小报主要是seo领域中视频需求，广泛服务于自媒体等项目，主要功能包括视频剪辑（足球/篮球）

<details open>
<summary>主要特性</summary>

- **便捷**     
    不需要改动原始mmaction2的代码

</details>

## datasets
#### 动作识别数据集
##### kinetics-400/600/800
包括400个类别，全部文件135g
#####  [tiny-kinetics-400](https://github.com/Tramac/tiny-kinetics-400)

#### 时序定位数据集


## apps
#### football
支持足球的裁剪，识别7个动作：进球、角球、任意球、黄牌、红牌、换人、界外球，还有一个背景类，识别上大部分识别成背景类






## Tips
- **数据准备**         
    小报支持两种数据类型：原始帧和视频，处理帧速度快，但是要大量空间，视频版本能节省空间，但是必须视频解码，
    算力开销很大。  

- **configs的命名风格**   
    {model}_[model_setting]_{backbone}_[misc]_{data_setting}_[gpu x batch_per_gpu]_{schedule}_{dataset}_{modality}    
    model:模型类型，如tsn
    model_setting：模型上的特殊设置,clip_len=1, frame_interval=1, num_clips=8,相当于把一段视频等分8份，在每一份的视频中随机抽取帧，总能抽8段帧
    backbone：主干网络类型，如r50
    misc：模型额外设置或插件，如dense，320p，video
    data_setting:采帧数据格式，形如{clip_len}x{frame_interval}x{num_clips}
    gpu x batch_per_gpu:gpu数量以及每个gpu上的采样
    schedule：训练策略设置，如20e表示20个epoch
    dataset: 数据集名，如kinetics400
    modality：帧的模态，如rgb，flow

- **线上训练报错**      
    ModuleNotFoundError: No module named '_lzma'??     
    sudo yum install xz-devel -y    
    sudo yum install python-backports-lzma -y    
    将requirements下的lzma.py cp到/usr/local/python3/lib/python3.6下，并安装pip install backports.lzma    

- **多卡训练**    
    apis/train.py 中116行中更换mm的训练
    python -m torch.distributed.launch   --nproc_per_node=8   --nnodes=1 --node_rank=0     --master_addr=localhost   --master_port=22222 	train.py        
    在k80上，dist_params = dict(backend='gloo')   
    
    python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_attetion_lstm  main.py  
    
- **yum源更新**
    ps -aux | grep yum
    sudo rm -rf /etc/yum.repos.d/*
    sudo wget http://10.96.126.170:8088/yum_for_suning_prd.repo -O /etc/yum.repos.d/yum_for_suning.repo
    sudo yum clean all
    sudo yum -y makecache
    sudo yum repolist all

- **mmcv_full**
    在k80上要重新编译mmcv_full，编译版本mmcv_full==1.4.4，cuda>=9.2,gcc>=5.4.0   
    pip install -r requirements/optional.txt
    MMCV_WITH_OPS=1 pip install -e . -v
    python .dev_scripts/check_installation.py
    https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html

- **gcc升级**
    sudo rpm -ivh gcc-5.4.0-1.el7.centos.x86_64.rpm
    export CC=/usr/local/bin/x86_64-unknown-linux-gnu-gcc
    export CXX=/usr/local/bin/x86_64-unknown-linux-gnu-g++

- **动作检测任务**    
    动作检测任务可以理解为动作定位+识别任务，需要在一个长视频中，先定位到一个动作发生的起止点，再识别出这个动作是什么
    <center><img src='https://ai-studio-static-online.cdn.bcebos.com/035726fa5f544e3d8ead9ae687db67fbfd28af11fab44c48adf2643b325748f0' width=600></center>
    <center>图2 动作检测任务流程</center>
    <br></br>

- **paddlepaddle**   
    ImportError: /usr/local/lib64/libstdc++.so.6: version 'GLIBCXX_3.4.22' not found  
    strings /usr/local/lib64/libstdc++.so.6 | grep GLIBC   
    ImportError: /usr/local/lib64/libstdc++.so.6: version 'CXXABI_1.3.11' not found
    rpm2cpio libstdc++-8.5.0-4.el8_5.x86_64.rpm | cpio -div
    //export LD_LIBRARY_PATH=/home/ivms/local_disk/usr/lib64 //就用这个方式
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sniss/local_disk/usr/lib64" >> ~/.bashrc
    source ~/.bashrc
    ImportError: /lib64/libc.so.6: version `GLIBC_2.18' not found //需要在5.4上编译glic，不然系统直崩溃
    sudo ln -s /usr/bin/_mv /usr/bin/mv
    tar -xvf glibc-2.18.tar.xz
    cd glibc-2.18
    mkdir build
    cd build    
    ../configure --prefix=/usr    
    make -j20
    make install
    cd /lib64
    ll | grep "libc."
    
    ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory //没啥用
    sudo yum whatprovides libstdc++.so.6
    sudo yum install libstdc++-4.8.5-44.el7.i686
    sudo yum remove libstdc++-4.8.5-44.el7.i686
    
    OSError: (External)  Cuda error(35), CUDA driver version is insufficient for CUDA runtime version.
    升级cuda10.1
    unzip cuda-10-1.zip -d $HOME/local_disk/
    sudo rm -rf /usr/local/cuda
    sudo ln -s $HOME/local_disk/cuda-10.1  /usr/local/cuda
    echo "export PATH=$PATH:/usr/local/cuda/bin" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
    source ~/.bashrc
    sudo pip install paddlepaddle_gpu-2.1.2.post101-cp36-cp36m-linux_x86_64.whl //paddlepaddle_gpu==2.0.2默认装10.2
    sudo pip uninstall six
    sudo pip install six
    
- **环境配置流程**  
    1.pip环境安装
    2.升级gcc
    3.升级cuda
    4.paddle-bfloat/paddlepaddle_gpu //其实不需要更新glic之类的
    5.编译ffmpeg // 
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sniss/local_disk/online/bin" >> ~/.bashrc
    source ~/.bashrc
    6.find /home/sniss/local_disk -name "*.ipynb_checkpoints" | xargs rm -rf

- **剪辑全流程输出**
    由于剪辑是先抽帧提取特征，bmn之后index就乱了，复原原图
    1.不抽帧，全video输入
    2.抽帧，bmn裁剪抽帧的视频，最后再转回来，考虑音频消失的问题
    

  
    
  
    
  
    
## ai+sports
#### 足球
背景、进球、角球、任意球、黄牌、红牌、换人、界外球

#### 乒乓球



## 










