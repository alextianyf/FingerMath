# 07 · 单数字手写识别：文献调研与研究方案

目标：在游戏真实的使用条件下（手指、鼠标或触控笔在符纸上写一个 0–9 的数字，模型在浏览器里运行），把单个数字的识别准确率再往上提。

本文分两部分：

1. 文献调研：主流做法是怎么解决这个问题的，各自的优缺点是什么。
2. 我们的方案：结合游戏的具体条件，下一步做哪些实验、怎么衡量。每个实验对应一个 notebook。

参考文献在文末，正文用 [编号] 引用，方便以后写文章直接复用。

---

## 0. 我们的问题有什么特殊之处

在讨论文献之前，先把我们的场景和"标准的 MNIST 问题"的差别说清楚。这些差别决定了哪些文献方法对我们有用。

| | MNIST 式的标准问题 | 我们的游戏 |
|---|---|---|
| 输入 | 扫描的纸面字迹（离线图像） | 屏幕上的笔画轨迹：每个点的坐标和时间都有（**在线**数据），我们再把它画成图 |
| 书写工具 | 纸和笔 | 手指（手机）、鼠标（电脑）、触控笔（平板） |
| 书写者 | 训练集和测试集的书写者不同 | 完全陌生的玩家，而且同一个玩家会写成百上千次 |
| 运行环境 | 不限 | 浏览器里，要求模型小（几百 KB）、快（约 10 毫秒以内） |
| 额外信息 | 无 | **屏幕上有哪些答案是已知的**（只有两三个候选），玩家打中妖怪等于告诉我们"这个字就是这个数" |
| 错误的代价 | 对称 | 不对称："读错成另一个在场的答案"最糟糕；"没看清，请再写一次"代价小得多 |

**书写工具本身是一个变量。**同一个人用鼠标、手指、触控笔写出来的字并不相同：

- **鼠标**：手腕和手臂在动，笔画更直、更有棱角，速度慢，抖动多；书写时常常省掉装饰性的笔画。比如带钩、带底座的"1"，用笔或手指写时可能出现，用鼠标写时就未必。
- **手指**：在小屏幕上写，笔画粗、比例随意，起笔和收笔常有拖尾。
- **触控笔**：最接近纸上书写，细节最多。

所以"书写者"和"书写工具"是两个需要**分开**考虑的维度：评估要按工具分开报告；训练数据要覆盖各种工具；个性化也要按"这个人、这个工具"来做。公开数据集里有触控笔（Pendigits [6]）和手指（Corr [7]），**没有鼠标**。鼠标数据只能自己采集，这也是采集环节必不可少的原因。

已有的工作（notebook 01–06，见 `06_recognition.md`）已经做到：

- 把 MNIST 上的 MLP 换成一个小 CNN（107k 参数，float16 后 223 KB）。
- 在**合成的**"未见过的书写者"测试集 `pad_tmpl` 上，准确率从 92.9% 提高到 96.9%。
- 剩下的错误主要是带钩、带底座的"1"被读成 7 或 2。

但有一个根本的空白：**所有测试集都是我们自己合成的，至今没有用任何真实的手指或鼠标笔迹量过。**合成数据的写法是我们自己设计的，它能告诉我们"在我们想到的写法上表现如何"，但不能告诉我们"在真实玩家身上表现如何"。下面的文献调研和方案，很大程度上就是围绕这个空白展开的。

---

## 1. 文献调研：主流做法

### 1.1 离线图像分类：MNIST 路线

**做法。**把字迹看成一张小图（28×28），用分类器识别。LeCun 等人 [1] 建立了 MNIST 基准，并用卷积网络（LeNet-5）证明了 CNN 在手写数字上的优势。Simard 等人 [2] 总结了几条实用经验：训练数据量最重要，用**弹性形变**（elastic distortion）等方式扩充数据最有效；CNN 明显优于全连接网络；简单的 CNN 不需要复杂的技巧。Cireşan 等人 [3] 把多个在不同预处理上训练的深度 CNN 取平均（multi-column），在 MNIST 上第一次接近人类水平。

**优点。**
- 方法成熟，数据现成（MNIST 有 7 万张；EMNIST [4] 从 NIST SD-19 扩展出更多数字和字母，格式和 MNIST 相同）。
- 对笔画顺序、书写方向完全不敏感：不管怎么写，只要最后的样子对就行。
- 我们现在上线的就是这条路线。

**缺点。**
- **领域差异（domain shift）**：MNIST 是 1990 年代美国人口普查员和高中生在纸上写、再扫描得到的。手指在玻璃上写的字更粗、更歪、比例不同，还有很多 MNIST 里罕见的写法（比如带钩的 1、欧洲式的 7）。
- MNIST 上的高分并不代表鲁棒。MNIST-C [5] 给 MNIST 测试集加了 15 种常见的扰动，一个在干净数据上 99.2% 的 CNN 平均只剩 91.2%。我们自己的 notebook 01 也看到了类似现象：MLP 在 MNIST 上 98.3%，换成游戏笔画风格的数据就掉到 93% 左右。
- 把轨迹画成图会**丢掉信息**：笔画的顺序、方向和速度都没了。比如"1"和"7"的差别有时就在第一笔是不是先往右走。

### 1.2 数据增强与合成数据

**做法。**用变换扩充训练数据：仿射变换（旋转、缩放、平移、错切）、弹性形变 [2]、笔画粗细变化；也可以用生成模型"写"出新的字，比如 Graves [8] 用 LSTM 生成手写轨迹，Ha 和 Eck [9] 的 sketch-rnn 生成简笔画轨迹。训练技巧上，label smoothing [10] 和 mixup [11] 能改善泛化和置信度。

**优点。**
- 成本低，不需要额外收集数据。
- 能覆盖"笔宽、大小、倾斜"这类**连续变化**的差异。我们在 notebook 06 里用"用游戏的笔重新画 MNIST 骨架"，把 `pad_skel` 提高了约 4 个百分点。

**缺点。**
- 增强只能覆盖我们**想得到**的变化。真实玩家的写法差异很多是**离散的**（带不带钩、7 中间加不加横、4 是开口还是闭口），这些靠随机变换造不出来。
- 合成数据和测试数据如果来自同一个生成器，测出来的分数会偏高。notebook 06 的结论里专门提到：带钩的 1 是我们设计进测试集的写法，不能拿来训练，否则测试集就"见过"了。

### 1.3 在线（轨迹）识别

**做法。**直接用笔画轨迹：一串带时间的 (x, y) 点，常常再加上笔画起落标记。经典的数据集有 UCI Pendigits [6]：44 个人在 WACOM 数位板上写的 10,992 个数字，公开版把每个数字重采样成 8 个点。与我们最接近的是 Corr 等人 [7] 的数据集：260 人在 iPhone 6 上**用手指**写的约 2 万个数字，保存了每个触点的坐标和时间戳，按书写者分开存储。他们用 2D CNN 识别画出来的图，准确率 98.5%；用 1D CNN 识别轨迹向量，准确率 95.9%。Google 的多语种在线手写识别 [12] 用 LSTM 加 Bézier 曲线编码轨迹，替代了旧的"先切分再解码"系统，错误率相对降低 20–40%，速度最多快 10 倍。综述见 Plamondon 和 Srihari [13]。

**优点。**
- 用上了图像里没有的信息：笔画顺序、方向、起笔和收笔的位置。
- 对"笔画粗细、画布分辨率"这些渲染相关的差异天然不敏感。
- 输入很小（几十个点），模型也可以很小。

**缺点。**
- 对**笔画顺序和方向**敏感：同一个"4"可以一笔写，也可以两笔写；"0"可以顺时针，也可以逆时针。这些差异需要足够的训练数据来覆盖。
- 训练数据少：MNIST 是离线图像，没有轨迹。公开的在线数字数据集规模小得多。
- 单用轨迹，准确率不一定比图像高：在 Corr 等人 [7] 的数据上，轨迹模型 95.9%，反而低于图像模型的 98.5%。

### 1.4 在线与离线融合

**做法。**同一个字，用轨迹模型和图像模型各识别一次，再把两个结果合并（比如概率相乘或加权平均）。Vinciarelli 和 Perrone [14] 在单个数字上做了这件事：在线识别器加上离线识别器后，错误率下降了 43%。

**优点。**
- 两种模型看的是不同的信息，犯的错误往往不一样，合并起来效果通常好于任何一个单独的模型。
- 我们的输入本来就同时有轨迹和图像，**不需要任何额外的数据采集**。

**缺点。**
- 要跑两个模型：下载体积和推理时间都会增加。
- 两个模型的置信度刻度不一定一致，合并前需要校准（见 1.6）。

### 1.5 书写者适应与个性化

**做法。**用某个书写者自己的样本调整模型。早期的在线手写系统就有书写者适应 [15]：先识别出这个人的"写法风格"，再用它调整分类器。深度学习时代的做法有两类：一类是微调最后几层；另一类是**原型网络** [16]，把每个类别的少量样本编码后取平均作为"原型"，新样本归到最近的原型。还有一类非常轻量的模板匹配方法：$1 [17] 和 $P [18] 识别器只用每类一两个样本、约 100 行代码就能识别笔画手势。

**优点。**
- 收益通常很大：我们的 notebook 04 用每个数字 5 个样本微调，在合成的书写者上把准确率从 90.8% 提到 97.2%。
- 能直接解决"某个人的特殊写法"（比如他的 1 总带钩）。数据增强解决不了这类问题。

**缺点。**
- 需要这个人自己的样本，也就是要有一个"校准"环节，或者在游戏中悄悄地收集。
- 适应过头会"过拟合"到这个人，或者被错误的样本带偏。

**我们的特殊优势**：游戏本身就会产生带标签的样本。玩家写下数字、AI 识别、妖怪被打中，这个字的真实标签基本可以确定，因为它就是那只妖怪的答案。这相当于一路玩、一路免费得到个性化数据，玩家不需要专门去"校准"。

### 1.6 置信度校准与拒识

**做法。**现代神经网络的输出概率往往过于自信，**温度缩放**（temperature scaling）[19] 用一个参数就能把概率校准得比较准。有了可靠的概率，就可以按 Chow 的规则 [20] 在"出错"和"拒识"之间做最优取舍：当最高概率低于某个阈值时，宁可说"没看清，再写一次"，也不要硬猜。label smoothing [10] 会改变置信度的刻度，用它时要注意这一点（notebook 06 已经提到）。

**优点。**
- 不改模型，只改决策，成本几乎为零。
- 正好对应我们"错误代价不对称"的场景：读错成另一个在场答案的代价，远大于让玩家再写一次。

**缺点。**
- 拒识太多会让玩家烦躁，阈值需要在真实数据上调。
- 校准本身也需要一份有代表性的数据，用合成数据校准，到了真实玩家身上可能又不准。

### 1.7 测试时适应

**做法。**部署后，用没有标签的测试数据继续调整模型。例如 Tent [21] 只更新归一化层的少量参数，让模型在新数据上的预测更"确定"（熵最小化）。领域适应的经典方法 DANN [22] 则在训练时就让模型学到和领域无关的特征。

**优点。**不需要标签，能跟着真实数据的分布走。

**缺点。**
- Tent 要按批（batch）更新，我们一次只有一个字，不太适用。
- 没有标签时，模型可能越调越偏，自己强化自己的错误。

对我们来说，1.5 节说的"游戏确认的标签"比无监督的测试时适应更可靠，所以这一类方法只作为参考。

### 1.8 上下文：用候选答案缩小范围

**做法。**识别时结合先验知识。在文本识别中，这个角色由语言模型或词典承担；在我们的游戏里，就是"屏幕上只有这几个答案"。我们现在的 `decide()` 已经在做这件事：像 8 的 3，只要场上有 3 就算 3。

**优点。**把十选一变成二三选一，难度大幅下降，这也是游戏实际接受率能到 98% 的原因。

**缺点。**
- 目前的规则是手工阈值，没有用一个统一的概率框架来推导。
- 它依赖模型的概率是否可靠（1.6），而这一点目前只在合成数据上验证过。

### 1.9 部署约束：小、快、在浏览器里

知识蒸馏 [23] 可以把大模型或集成模型的能力压缩进小模型；量化 [24] 可以进一步减小体积、加快推理。Cireşan 等人 [3] 那种多模型集成虽然准，但我们的浏览器预算放不下，所以只能用蒸馏的方式间接利用。

---

## 2. 各路线的总结

| 路线 | 核心思路 | 对我们的主要优点 | 对我们的主要问题 | 优先级 |
|---|---|---|---|---|
| 离线图像 CNN [1][2][3] | 把字画成图再分类 | 成熟，已上线 | 领域差异；丢掉笔画顺序信息 | 已有基础 |
| 数据增强 [2][8][10][11] | 扩充训练数据 | 便宜 | 造不出离散的写法差异 | 继续使用 |
| **真实在线数据** [6][7] | 用真实的笔、手指数据训练和测试 | 直接解决"没有真实测试集"的问题 | 数据量小；需要注意许可 | **最高** |
| 在线轨迹模型 [7][12] | 用轨迹而不是图像 | 用上方向、顺序信息 | 单独用不一定更好 | 高 |
| **在线 + 离线融合** [14] | 两个模型合并 | 文献中错误率 −43%；我们的输入天然支持 | 两个模型的成本 | **高** |
| 校准与拒识 [19][20] | 让概率可靠，按代价决策 | 零成本，符合不对称代价 | 需要真实数据校准 | 高 |
| 个性化 [15][16][17][18] | 用玩家自己的样本调整 | 收益大；游戏自带标签 | 需要设计更新机制、防止带偏 | 中（第二阶段） |
| 测试时适应 [21][22] | 无标签自适应 | 无需标签 | 单样本不适用；容易漂移 | 低 |
| 蒸馏、量化 [23][24] | 压缩模型 | 满足浏览器预算 | 工程工作量 | 需要时再用 |

---

## 3. 我们的方案

### 3.1 核心判断

1. **最大的瓶颈是评估，不是模型。**我们在合成数据上已经到 97%，但不知道真实玩家是多少。没有真实测试集之前，任何"改进"都可能只是在拟合我们自己的合成器。所以第一步是建立**真实的、按书写者划分的**测试集。
2. **我们的数据本来就是在线的**，这是 MNIST 路线没有利用的信息。图像 CNN 加轨迹模型的融合，是文献里有明确收益、又不需要新数据的方向。
3. **决策层还有空间**：用校准过的概率加上候选答案的先验，按"读错比让玩家重写代价大得多"来推导 `decide()` 的规则，而不是手工设阈值。
4. **个性化是长期最有效的手段**，而且游戏能免费提供标签，但放在第二阶段：先把通用模型和评估做扎实。

### 3.2 数据

| 数据 | 内容 | 许可 | 用途 |
|---|---|---|---|
| MNIST [1] | 7 万张离线图像 | CC BY-SA 3.0 | 训练（已在用） |
| UCI Pendigits [6]（`-orig` 原始轨迹） | 44 人，数位板，10,992 个；按书写者分为训练 30 人、测试 14 人 | CC BY 4.0 | 训练和测试都可以 |
| Corr 等人的 iPhone 手指数字 [7] | 260 人，手指书写，约 2 万个，带时间戳和书写者编号 | 仓库里**没有写许可证**，只要求引用 | 建议**只用作测试集**，见 3.4 |
| 我们自己采集 | 用 `collect/index.html` 在电脑（鼠标）和手机（手指）上采集 | 自有 | 最终的、最接近游戏的测试集 |
| 已有合成集 `pad_skel`、`pad_tmpl` | 见 notebook 01、06 | 自有 | 保留，用于和以前的结果对比 |

所有真实轨迹都通过**游戏自己的渲染和预处理流程**变成 28×28 图像，保证训练、测试和游戏里看到的是同一种图。

### 3.3 实验计划（每一步一个 notebook）

编号接着已有的 01–06。

| Notebook | 内容 | 产出 |
|---|---|---|
| **07_real_benchmarks** | 下载并解析 Pendigits 原始轨迹和 Corr 数据集；把轨迹按游戏的笔和预处理画成图；**按书写者**划分训练、验证、测试集；可视化样本 | 真实测试集，以及数据说明（多少人、多少样本、每个数字多少） |
| **08_baseline_on_real** | 用真实测试集评估已上线的 CNN 和旧 MLP：总体和每个数字的准确率、混淆矩阵、每个书写者的准确率分布、置信度校准（ECE、可靠性图）、"游戏实际接受率"和"打错妖怪率" | 第一次知道真实玩家大概是什么水平；剩下的错误到底是什么 |
| **09_train_with_real** | 把 Pendigits 训练部分加入训练（Corr 只评估，不训练），并加入针对所有数字的倾斜、笔宽增强；4 个方案 × 2 个随机种子的消融实验 | 新的图像 CNN，每个改动的贡献 |
| **10_error_audit** | 逐个查看剩下的错误样本，分为"模型确实读错""人眼也认不出""疑似标签错误"三类；先由我初筛，再由你确认 | 真实数据上的准确率上限，以及还有多少可以改进的空间 |
| **11_trajectory_and_fusion** | 轨迹模型（1D 卷积或小 GRU，输入为重采样后的点和方向）；与图像 CNN 做晚期融合（加权平均对数概率）；比较单独和融合的效果 | 融合是否值得做，以及在浏览器里的体积和延迟 |
| **12_architecture** | 固定使用 09 选出的数据，公平比较不同架构：现有 CNN、加宽或加残差的 CNN [25]、深度可分离卷积的轻量 CNN [26]、多个模型集成 [27] 后蒸馏 [23] 到小模型、11 的融合模型；空间变换网络 [28] 作为可选项。每个都报告准确率、模型大小和浏览器推理时间 | 架构能带来多少收益；如果收益很小，这本身就是结论 |
| **13_calibration_and_decision** | 温度缩放 [19]；把 `decide()` 写成"概率 × 候选先验 + 错误代价"的决策规则；画出错误–拒识曲线 [20]；用模拟的游戏场景比较新旧规则 | 新的 `decide()` 规则及其参数 |
| **14_personalisation**（第二阶段） | 模拟"游戏确认的标签"：每个书写者按时间顺序逐个出现样本，用原型 [16] 或最后一层微调在线更新，按"人 + 工具"分别适应；测量随样本数增加的准确率，以及错误标签的影响 | 是否、以及如何在游戏里做无感个性化 |
| **收尾** | 选定上线方案，导出到浏览器，用游戏自己的 JS 做端到端验证，更新 README 和研究报告 | 上线 |

**关于架构的判断（2026-10-27 补充）。**单纯把 CNN 换大、换新，预期收益很小：notebook 08 显示剩下的错误主要来自训练数据里没见过的写法（数据问题），以及本身模糊或疑似标错的样本（任何模型都救不回来，见 [29]）；在 MNIST 类问题上小 CNN 已接近上限。所以架构研究的重点是**利用新信息**（11 的轨迹分支）和**改变成本结构**（12 的集成后蒸馏：只增加训练成本，上线体积不变），并且排在 10 的错误审查之后，先弄清还剩多少可改进的空间。Transformer、胶囊网络等需要大量数据的结构不在考虑范围内。

### 3.4 需要你决定的事

**Corr 等人的 iPhone 数据集没有明确的许可证。**它是和我们最接近的真实数据（手机、手指），我的建议是：

- **只把它用作测试集**，不用来训练要上线的模型。这样不存在"把别人的数据打包进产品"的问题，引用他们的论文即可。
- 训练只用许可明确的数据：MNIST、Pendigits（CC BY 4.0）、我们自己的合成数据和自己采集的数据。

另外，`collect/index.html` 采集真实样本需要你本人和身边的人参与。建议**电脑鼠标和手机手指各采一批**，每人每个数字写 3–5 次；最好**同一个人两种工具都写**，这样才能直接比较"同一个人、不同工具"的差异。这一批会成为最终、最可信的测试集，也是**唯一的鼠标数据来源**。这一步不急，可以和 07–08 同时进行。

**已决定（2026-10-27）：**Corr 数据集只用作测试集。

### 3.5 评价指标

主要指标都在**真实数据、按书写者划分**的测试集上计算，并且**按书写工具（触控笔、手指、鼠标）分开报告**：

1. 十选一的准确率（总体和每个数字）。
2. 游戏实际接受率：屏幕上有 2–3 个候选答案时，被判为"正确命中"的比例。
3. **打错妖怪率**：被判成另一个在场答案的比例。这是最影响体验的错误，要求不能变差。
4. 拒识率："没看清，请再写一次"的比例。
5. 部署成本：模型大小（目标 < 500 KB）和浏览器端一次完整识别的时间（目标 < 15 毫秒）。

### 3.6 风险

- **Pendigits 的采样较粗**：每 100 毫秒一个点 [6]，画出来的笔画会比实际的折线更直。要在 07 里可视化检查，必要时用样条插值。
- **数位板和手指的差异**：Pendigits 用的是触控笔，和手指不同。这也是为什么 Corr 数据集和我们自己采集的数据很重要。
- **测试集被"用旧"**：反复在同一个测试集上挑方案会过拟合它。和 notebook 06 一样，方案按验证集挑选，测试集只在最后报告。

---

## 参考文献

[1] Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

[2] P. Y. Simard, D. Steinkraus, J. C. Platt. Best practices for convolutional neural networks applied to visual document analysis. *Proc. 7th International Conference on Document Analysis and Recognition (ICDAR)*, 2003. doi:10.1109/ICDAR.2003.1227801

[3] D. Cireşan, U. Meier, J. Schmidhuber. Multi-column deep neural networks for image classification. *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3642–3649, 2012. doi:10.1109/CVPR.2012.6248110. arXiv:1202.2745

[4] G. Cohen, S. Afshar, J. Tapson, A. van Schaik. EMNIST: an extension of MNIST to handwritten letters. *Proc. International Joint Conference on Neural Networks (IJCNN)*, 2017. arXiv:1702.05373

[5] N. Mu, J. Gilmer. MNIST-C: A robustness benchmark for computer vision. *ICML Workshop on Uncertainty and Robustness in Deep Learning*, 2019. arXiv:1906.02337

[6] E. Alpaydin, F. Alimoglu. Pen-Based Recognition of Handwritten Digits. UCI Machine Learning Repository, 1996. doi:10.24432/C5MG6K. https://archive.ics.uci.edu/dataset/81 （CC BY 4.0）

[7] P. J. Corr, G. C. M. Silvestre, C. J. Bleakley. Open source dataset and deep learning models for online digit gesture recognition on touchscreens. *Irish Machine Vision and Image Processing Conference (IMVIP)*, 2017. arXiv:1709.06871. 数据：https://github.com/PhilipCorr/numeral-gesture-dataset

[8] A. Graves. Generating sequences with recurrent neural networks. arXiv:1308.0850, 2013.

[9] D. Ha, D. Eck. A neural representation of sketch drawings. *International Conference on Learning Representations (ICLR)*, 2018. arXiv:1704.03477

[10] R. Müller, S. Kornblith, G. Hinton. When does label smoothing help? *Advances in Neural Information Processing Systems (NeurIPS)*, 2019. arXiv:1906.02629

[11] H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz. mixup: Beyond empirical risk minimization. *ICLR*, 2018. arXiv:1710.09412

[12] V. Carbune, P. Gonnet, T. Deselaers, H. A. Rowley, A. Daryin, M. Calvo, L.-L. Wang, D. Keysers, S. Feuz, P. Gervais. Fast multi-language LSTM-based online handwriting recognition. *International Journal on Document Analysis and Recognition (IJDAR)*, 23:89–102, 2020. arXiv:1902.10525

[13] R. Plamondon, S. N. Srihari. On-line and off-line handwriting recognition: a comprehensive survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(1):63–84, 2000.

[14] A. Vinciarelli, M. P. Perrone. Combining online and offline handwriting recognition. *Proc. 7th ICDAR*, pp. 844–848, 2003. doi:10.1109/ICDAR.2003.1227781

[15] S. D. Connell, A. K. Jain. Writer adaptation of online handwritten models. *Proc. 5th ICDAR*, pp. 434–437, 1999.

[16] J. Snell, K. Swersky, R. Zemel. Prototypical networks for few-shot learning. *NeurIPS*, 2017. arXiv:1703.05175

[17] J. O. Wobbrock, A. D. Wilson, Y. Li. Gestures without libraries, toolkits or training: a $1 recognizer for user interface prototypes. *Proc. ACM Symposium on User Interface Software and Technology (UIST)*, pp. 159–168, 2007.

[18] R.-D. Vatavu, L. Anthony, J. O. Wobbrock. Gestures as point clouds: a $P recognizer for user interface prototypes. *Proc. ACM International Conference on Multimodal Interaction (ICMI)*, pp. 273–280, 2012.

[19] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On calibration of modern neural networks. *Proc. International Conference on Machine Learning (ICML)*, 2017. arXiv:1706.04599

[20] C. K. Chow. On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1):41–46, 1970.

[21] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, T. Darrell. Tent: fully test-time adaptation by entropy minimization. *ICLR*, 2021. arXiv:2006.10726

[22] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, V. Lempitsky. Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59):1–35, 2016.

[23] G. Hinton, O. Vinyals, J. Dean. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.

[24] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, D. Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR*, 2018. arXiv:1712.05877

[25] K. He, X. Zhang, S. Ren, J. Sun. Deep residual learning for image recognition. *CVPR*, 2016. arXiv:1512.03385

[26] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, H. Adam. MobileNets: efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861, 2017.

[27] B. Lakshminarayanan, A. Pritzel, C. Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*, 2017. arXiv:1612.01474

[28] M. Jaderberg, K. Simonyan, A. Zisserman, K. Kavukcuoglu. Spatial transformer networks. *NeurIPS*, 2015. arXiv:1506.02025

[29] C. G. Northcutt, A. Athalye, J. Mueller. Pervasive label errors in test sets destabilize machine learning benchmarks. *NeurIPS Datasets and Benchmarks Track*, 2021. arXiv:2103.14749

说明：[1]–[7]、[12]–[14]、[16]–[18]、[20]、[21] 的出处已在检索时逐条核对过（作者、年份、会议或期刊）；[25]–[29] 为 2026-10-27 补充的标准文献；其余为本领域的标准文献，正式引用前建议再对照一次原文。
