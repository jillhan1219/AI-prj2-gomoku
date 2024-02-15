## 五子棋
### 代码结构
```sh
.
├── DQN
│   ├── DQN.py
│   ├── GomokuBoard.py
│   ├── algorithm.py
│   ├── checkpoints
│   │   ├── checkpoint
│   │   ├── model.ckpt-42000.data-00000-of-00001
│   │   ├── model.ckpt-42000.index
│   │   ├── model.ckpt-42000.meta
│   ├── img
│   └── main.py
├── README.md
├── alphabeta-prunning
│   ├── algorithm.py
│   ├── alpha_beta_pruning.py
│   ├── eval_fn.py
│   ├── img
│   └── main.py
└── alphazero
    ├── algorithm.py
    ├── game_board.py
    ├── img
    ├── main.py
    ├── mcts_alphaZero.py
    ├── mcts_pure.py
    ├── model_15_15_5
    │   ├── best_policy.model.data-00000-of-00001
    │   ├── best_policy.model.index
    │   ├── checkpoint
    │   └── readme.md
    ├── policy_value_net_tensorlayer.py
    └── train.py
```
### 测试及运行
- 在每个文件夹下运行
```sh
python3 main.py
```
- 环境
  - alphabeta-prunning：基础环境
  - DQN：tensorflow 2.6
  - alphazero：tensorflow 1.15