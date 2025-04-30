训练：
    python train.py --task=cowa_ppo 
    开启训练后，建议将/home/zhengde.ma/humanoid-gym-main/humanoid/envs/custom目录下的，env和cfg文件拷到新产生的log文件中，以便找到当时的参数

评估：
    log文件在/home/zhengde.ma/humanoid-gym-main/logs/Cowa_ppo
    评估前，将.pt文件拷贝在success文件下，执行：python play.py
    评估后，会产生video文件，在/home/zhengde.ma/humanoid-gym-main/videos/Cowa_ppo目录下，每次play都会保存