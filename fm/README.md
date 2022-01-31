# FM实操

> 2021-08-22



## 数据格式

user_id,item_id,category_id,behavior_type,timestamp

```bash
$ head data_1.csv
1,2268318,2520377,pv,1511544070
1,2333346,2520771,pv,1511561733
1,2576651,149192,pv,1511572885
1,3830808,4181361,pv,1511593493
1,4365585,2520377,pv,1511596146
1,4606018,2735466,pv,1511616481
1,230380,411153,pv,1511644942
1,3827899,2920476,pv,1511713473
1,3745169,2891509,pv,1511725471
1,1531036,2920476,pv,1511733732
```



## 思路

负样本：pv

正样本：buy

使用FM，评分矩阵，pv为0，buy为1



## 文件说明

- data_1.csv

  原始数据集

- data_analysis 

  统计原始样本中共有多少个user和item，以及id的取值范围。用于确定fm的超参数M和K
  
  ```bash
  $ wc data_analysis/item_id 
  629096  629096 4897519 data_analysis/item_id
  
  $ wc data_analysis/user_id 
  19544   19544  138581 data_analysis/user_id
  ```

- fm_model.py

  fm模型

- process.py

  原始样本数据处理为fm的数据输入格式（train_data数据文件）

  python3 process.py > train_data

- score

  即为模型对于test集的预估得分，后续用于计算auc

  python3 fm_model.py train_data > score

- label

  测试集，用于计算auc

  tail -n -40000 train_data > label

- auc.py

  计算二分类模型的auc

  python3 auc.py
