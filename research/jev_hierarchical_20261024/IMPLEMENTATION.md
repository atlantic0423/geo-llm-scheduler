# Jev 两档分层控制实现说明

## 当前状态

2026-10-24：用户要求实际实现 JEV_E05 / JEV_E09，并将发送给 Jev 的提示词全部使用英文，
中文仅作阅读。基于冻结 Flash 引擎 89f14655b27b6a5c4a49f6668705a5e090787089 建立独立研究分支。
默认算法与旧 E16 / Flash 源不修改。本研究机制效果待验证，不等于采用或确认有益。

## 两版实际差异

只改变 JevSettings.epsilon_g：0.5 或 0.9。动作、Job、phase/Region、路径/位置分别抽样：
以该概率选最大项（并列均匀），其余按 Jev 完整概率分布抽样。停止使用 Noul 阈值，不随机。

数学定义如下，M是当前最大概率选项集合：

$$
\pi(c)=\epsilon_g U_M(c)+(1-\epsilon_g)p_{\mathrm{Jev}}(c).
$$

这是 Jev 概率引导的 epsilon-greedy 变体，不是均匀探索的标准定义；随机分支仍可抽到最大项。
并列使用显式 RNG。动作流与每动作目标选择流独立，目标内部再按job/phase/Region/path/position用途派生随机流，均来自 master seed；
每次层级选择都产生独立抽样，下一请求使用实际抽中父对象。不是整条轨迹一次掷硬币。
所有字段相同的 arms.json 由脚本导出，测试只允许 epsilon_g 不同。

## 逐步发送什么

完整英文原文、中文翻译、三个偏好组的实际动作问题和每个类型的问题见
[PROMPTS_EN_ZH.md](PROMPTS_EN_ZH.md)。它从运行代码常量生成，测试逐字检查，无人工另写英文版本。
源码 jev_prompts.py 的 BACKGROUND_EN 和 PROMPTS 的英文项是唯一发送来源。
BACKGROUND_ZH 和中文项只被文档导出器使用，不进入运行请求。

每次 state 均有背景、真实拓扑与电价、动作历史和预算。对象请求另带当前解、
细权重、全部 Job 和全部地区窗口功率。为减小请求而不丢工件，工序明细使用列定义加行数组，
保留完整精度和全部行；名字不发送，以数值 ID 表示，不靠名字语义选工件。
候选 criteria 由代码实时填充完整合资格选项，不是固定几个示例标签。
所有输入 construction、序列化、网络和解析均在生产墙钟内；50%等待守卫是明确研究参数。

动作在初始化、约总预算三分之一/三分之二安全边界更新，每次一次POST批量三组 Choice；
迟到检查点合并，不补发。动作分布组内共享，真实接受仍用细权重。
原 strict/delta0 门通过后执行至多L5，每步最多B6。每步候选共享冻结 incumbent，
生成完才进入 exact，不把本批已评结果提前喂给后续候选选择。
候选不足、重复或失败不强行补满，也不当成模型已经优化成功。

## A1–A8 的实现边界

- A1：全部有合法替代实例的Job，再选phase与同地区实例。
- A2：全部有合法不同同地区路径的Job，再选路径。
- A3：全部有合法跨地区路径的Job，再选地区和P/D路径。
- A4：保留正等待且可前插的资格，在全部合资格Job/phase/位置中选择，删除启发式top-B目标限制。
- A5：保留P+D总等待为正和两者前移的资格，先选P位置再选条件D位置；不物化二次方全部位置对。
- A6：对全部Job逐个无放回选择约10%且至少2个destroy工件；原随机合法repair不变，不能称Jev选择repair。
- A7：原packing_moves有限合法正压缩时刻全部参与，Jev选择Job/phase/time；跳过原两代表/2B预筛是研究变化。
- A8：一Region，singleton先行或原support正边际组，成员上限、precedence-aware/窄窗优先、
  每成员至多6次位置检查、2B构造上限、原horizon、全部原并列峰及新峰严格gate保留。
  Jev选择Region、singleton Job/phase、support组与各次有限关键位置；不添加任意helper或回溯。
  A8时刻列表只保证前驱边界，资源仍由代码检查，失败rollback。

超过255选项按稳定ID区间分层，每次最多255项，全部原选项可达。区间首尾描述不是
对全体候选的全局评分，分层选择不等价于单次全局Choice。API不可用时回退为全eligible均匀，
分层回退按区间大小加权；这些都是有日志的回退，不能作为有效Jev决策统计。

SSGS、exact、同源best-of-B、全部可行exact的Archive机会、细权重接受、timing diagnostics、
两重点Polish与邻居自身细权重替换均复用已验收引擎。原engine/operator源文件未修改。

## 停止与异常

第1–4步真实exact/接受/Archive处理后，问“是否停止当前轨迹”。Noul严格大于0.5停止；
恰好0.5继续，第5步不再多问。这不是全局最优判定。
失败时连续两步无局部接受且无Archive新目标事件才触发显式本地fallback，否则继续，
始终受步数和生产时间限制。停止后Polish按原位置执行，不把其收益归入最后动作。

固定模型 jev-1.14.0；不用latest、不重试、不自动换模型。密钥从环境读取，
只有固定官方HTTPS子进程使用；stdin/request/evidence/Git都不写密钥。
父进程硬截止kill/reap。默认单run 512请求、单次15秒、最多50%总生产预算等待，
累计已计费或保守预留输入400万token；单次预留默认请求64k上界，未知失败保留预留。
这些是单run工程守卫，不是已核账号余额或多run全局预算。没有自动发起正式矩阵。

CLI在结束后对P100和Archive全部最终成员重新运行独立exact并要求逐字段一致；
失败不写“审计成功”的结果。requests.jsonl保存逐请求，result.json保存抽样、usage及最终结果。
当前公开CLI只运行指定一臂，不隐式启动对比campaign。

## 验证记录

- 专项测试已通过，含抽样分布、并列与RNG、200Job全候选、条件父选择、A1–A8独立exact、
  A8新峰/horizon、逐步Noul边界、API预算/超时/回退、英文wire和中英文文档一致性。
- 完整门禁、D03、replay、wheel/CLI及准确head双Python CI以本批最终验收回执为准。
- 测试使用确定性注入requester和模拟HTTPS，没有新的付费Jev调用；真实服务的当前延迟、
  上下文token量与调度质量均未由本批验证。
- 英文是用户选择和统一术语的工程安排，没有做中文/英文效果实验。

Notion研究协议：https://app.notion.com/p/4038878b8019816a9603d2b0634ccc5c

