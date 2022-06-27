# OGB-LPP
Link Property Prediction of OGBCancel changes

## **LRGA module augmented for link prediction**

The ogbl-collab  is a dateset for link prediction. The challenge leaderboard can be checked at: https://ogb.stanford.edu/docs/leader_linkprop/. We apply LRGA module augmented with GCN to solve this challenge and this repo contains our code submission. 

## **Requirements**

Install base packages: `Python==3.6 Pytorch==1.7.1 pytorch_geometric==2.0.1 ogb==1.3.2`

## Results on OGB Challenges

| ogbl-collab | (Hits@20)    | (Hits@50)       | (Hits@100)  |
| ----------- | ------------ | --------------- | ----------- |
| Validation  | 100.00± 0.00 | **100.00±0.00** | 100.00±0.00 |
| Test        | 60.38±0.64   | **69.09± 0.55** | 73.26±0.25  |

**## Reproduction of performance on OGBL**

*### ogbl-collab:** 

```shell
python main.py --data_name=ogbl-collab  --encoder=GCNWithAttention --runs=10 --predictor=DOT --use_valedges_as_input=True --year=2010 --train_on_subgraph=True --eval_last_best=True --dropout=0.3 --gnn_num_layers=1  --use_lr_decay=True --random_walk_augment=True --walk_length=10 --loss_func=WeightedHingeAUC --data_path='dataset/'
```

**## Reference**

[1] https://github.com/zhitao-wang/PLNLP

[2] https://github.com/omri1348/LRGA


