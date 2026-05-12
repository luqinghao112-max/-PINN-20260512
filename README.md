# 神经网络反演流程

本文件夹保存从理论数据整理、网络训练、单样本反演到批量验证的代码。  
主模型仍调用上一级目录中的 `S2_Singleprobe_Sampling_V14_compute_curve.m`，因为它是已经定稿的 MATLAB 正演内核，未做改名。

## 文件顺序

### 1. 生成理论数据

文件：`../1_make_data.m`

功能：

- 随机生成 2000 条均质 4 参数理论曲线
- 反演参数：`Kh, Kv, S, C`
- 已知参数：`q, mu, As, hw`

运行：

```matlab
run('1_make_data.m')
```

输出：

- `../data_2000.mat`

### 2. 整理训练数据

文件：`2_pack_data.py`

功能：

- 读取 `data_2000.mat`
- 整理成神经网络训练用的 `npz`
- 输入为 `log10(Kh), log10(Kv), S, log10(C), log10(t)`
- 输出为 `log10(Delta_p)`

运行：

```powershell
D:\ProgramData\anaconda3\envs\singleprobe_pinn\python.exe .\nn_flow\2_pack_data.py
```

输出：

- `data\train_data.npz`
- `data\meta.json`

### 3. 训练网络

文件：`3_train_net.py`

功能：

- 训练压力响应代理网络
- 按曲线划分训练集、验证集、测试集
- 默认划分：70% / 15% / 15%

运行：

```powershell
D:\ProgramData\anaconda3\envs\singleprobe_pinn\python.exe .\nn_flow\3_train_net.py --epochs 500 --print-every 50
```

输出：

- `model\net.pt`
- `model\norm.npz`
- `model\metrics.json`
- `fig\train_loss.png`

### 4. 检查物理核

文件：`4_check_core.py`

功能：

- 用 Python 物理核反演单条曲线
- 用于检查 Python 物理核和 MATLAB 正演内核是否一致
- 这一步不是神经网络训练，只是物理基准检查

运行：

```powershell
D:\ProgramData\anaconda3\envs\singleprobe_pinn\python.exe .\nn_flow\4_check_core.py --sample-idx 998 --max-nfev 80
```

输出：

- `core_out\result_998.json`
- `core_out\result_998.csv`
- `core_fig\fit_998.png`

### 5. 单条曲线网络反演

文件：`5_invert_one.py`

功能：

- 固定训练好的网络权重
- 只优化 `Kh, Kv, S, C`
- 可选把反演参数代回物理核复查

运行：

```powershell
D:\ProgramData\anaconda3\envs\singleprobe_pinn\python.exe .\nn_flow\5_invert_one.py --sample-idx 998 --epochs 800 --print-every 100 --phys-check
```

输出：

- `out\fit_998.json`
- `out\hist_998.csv`
- `out_fig\fit_998.png`

### 6. MATLAB 复查

文件：`../6_check_matlab.m`

功能：

- 将第 5 步反演得到的参数代回 MATLAB 正演内核
- 对比原始压力曲线和导数曲线

运行：

```matlab
run('6_check_matlab.m')
```

输出：

- `mat_out\summary.csv`
- `mat_fig\check_998.png`

### 7. 批量测试集验证

文件：`7_check_batch.py`

功能：

- 对测试集 300 条曲线做批量反演
- 统计参数误差和曲线误差
- 输出交会图、直方图、箱线图和样本误差图

运行：

```powershell
D:\ProgramData\anaconda3\envs\singleprobe_pinn\python.exe .\nn_flow\7_check_batch.py --split test --epochs 800 --print-every 100 --phys-check all --recheck-print-every 25
```

输出：

- `batch_out\results_test.csv`
- `batch_out\history_test.csv`
- `batch_out\summary_test.json`
- `batch_out\worst_test.csv`
- `batch_fig\param_test.png`
- `batch_fig\hist_test.png`
- `batch_fig\box_test.png`
- `batch_fig\quality_test.png`
- `batch_fig\sample_test.png`

## 推荐运行顺序

```text
1_make_data.m
2_pack_data.py
3_train_net.py
4_check_core.py
5_invert_one.py
6_check_matlab.m
7_check_batch.py
```

其中第 4 步是物理核检查，不是每次都必须跑；正式做网络结果时，重点看第 5、6、7 步。
