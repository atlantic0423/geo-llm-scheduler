# D3 v1：搜索前来源—算子—目标方向响应筛查

Material Passport：academic-research-suite / experiment-agent；2026-10-06；run/validate；未来机制与在线收益 UNVERIFIED。本协议在新标签生成前冻结，用户已授权直接验收、预跑与开跑。不是当前默认算法，不更改共享 Q、Trigger、Archive 或 A1–A8。

## 问题与范围

P2 已检验生成后的方向邻域替换。D3 检验给定原 caller 来源和固定 action，搜索开始前是否能选到更有价值的指导方向。当前 A1–A8 generator 不接收 weight；因此 v1 的指导只改变同一 B3 raw pool 的 best-of-B 保留，不能称改变构造、自动发现跨方向图或完整在线 HV 收益。

每源在 10%/50%/90% 的三个阶段、三类 caller preference 采九个面板。来源固定为 caller incumbent，目标是同一个出生邻域内按独立观测 RNG 抽到的六个 slot，包括重复表型和空调用，全部保留。八个 action 等权，不修改原 Q。每 action 仅构造一次原始 B3 pool；各指导方向共享原始 proposals、exact 结果和 updated-ideal selection context，按各自 weight 只在严格改善时保留最佳候选，否则返回来源。所有完整候选独立 exact 复验。

## 矩阵、独立性和资源

全新 48 base：50 jobs 895001–895024；100 jobs 905001–905024；各前16 base 开发、后8留出，合计32训练、16测试。每 base H/T、1101/2202/3303 三 seed，总288 source +288 paired probe。各规模 base 位置 mod 2 分主机，两电价及所有 seed 保持同主机；每台144对、108 source worker-hours。

50/100预算1800/3600秒；全216 source worker-hours，两个容器各8来源+2离线槽位，采样理想下界13.5小时，预计15–18小时，实际 Linux pilot 校准后给 ETA。不为达到某个显著性追加样本。pilot 每节点894001+node/904001+node，6606/7707/8808，120/240秒，12对/节点，不计正式统计。

## 标签与强对照

共同冻结 panel context 用于事后测量，selection 使用原 MacroSearch updated ideal。令目标 slot 为 $j$，用自己的权重和 incumbent 计算

$$
d_j(y)=\max\{0,g(p_j;\lambda_j,c)-g(y;\lambda_j,c)\}.
$$

同六-slot池的最多两次替换收益为最大两个 $d_j$ 之和除以六；路线标签是保留输出与未搜索来源的该值之差，可以为负。这是即时替换潜力，不是实际全局替换或 HV。

GEOMETRIC：最接近 caller weight 的 slot；RANDOM：固定独立 RNG 选 slot；RGAIN：在动作之前，用已知来源目标值对六目标计算即时 gain，最高优先，同分用几何距离和 index。BASE：每 action 一个 alpha=1 ridge，用来源/目标质量、权重、距离、目标 causal history/stagnation、阶段/偏好/规模、来源即时 gain 预测标签。RESPONSE：BASE 加十维廉价来源机会与目标 Flow weight 及其平方的20个交互。SHUFFLED：相同模型容量，但来源机会向量在 job/stage/preference/action 匹配的不同 panel 之间打乱，训练与测试分别处理。单 source 的六方向共享一个机会向量，不能在单池内打乱这一常量假装有效对照。

POST_RGAIN：动作完成后在同一完整候选池直接按共同六目标真实 gain 选最优，作为 oracle 上界；不能作为搜索前预测，也不能把预测方法偶然超过几何解释为超过该上界。输出必须不超过上界，否则审计失败。所有可复现 raw pools、选择上下文、路线身份、空调用与成本保留。六-slot与全20-neighbor/全局100-direction分开，不扩大新机制候选池。

## 冻结统计和停止门

只在两个正式 shard 全部通过原 validator 后联合训练；任何留出 base 不进入 scaler/fit/参数选择。每 action 固定 ridge，训练时相同完整面板/方向数使各 base 等权。所有阶段、action、tariff、seed先等权，再以16留出 base为独立单位。主比较 RESPONSE−BASE、RESPONSE−RGAIN、RESPONSE−SHUFFLED，20,000 base bootstrap、65,536 exact signflip、Holm三比较；每项均值至少0.001、95%CI下界大于0、Holm p小于.05才通过本轮机制筛查。几何/随机和 action/阶段/规模分层为描述性，不替换主门。

来源特征提取、原始构造/评价、失败尝试、模型 fit/预测、exact 审计分别记时。共用 pool 的离线成本与在线仅走一路的成本不同，不由该实验宣称在线节时。未通过则停止 v1 在线晋级，不普遍否定所有响应表征；通过后仍需独立在线等时间 F6/RGAIN/预测/打乱和 IGD+/中部覆盖护栏。新颖性须另与 MRLEA、matching、direction/operator joint selection 核查；本协议不宣布创新。

## 续跑与验收

冻结 source commit/canonical、协议和所有输入/初始化/配置 hashes。任务级续跑复用原 validator 全验真的完成件；同源/probe配对同主机，面板 checkpoint 的 source hash 与 spec/hash/host绑定；排他锁、stop.request、失败停止新派发和 quarantine 保留。跨主机未完成 pair须显式迁移整体重启，不能称恢复进程内部状态。不得自动重试真实算法失败，不删除新用户暂停请求。

正式前本地完整质量门、默认三算法固定代数回放、目标Linux短pilot原验真、fullraw回收/manifest/archive/source/Release回读核验。启动验收核对真实PID/命令、两次增长心跳及cgroup；完整正式结果与联合统计/MD/PR/Release为后续独立任务，启动不标记科学成功。
