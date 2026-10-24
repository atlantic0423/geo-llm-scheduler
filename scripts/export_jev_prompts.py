"""Generate bilingual prompt documentation from actual wire constants."""

import json
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.experiments.jev_decisions import JevSettings
from geo_llm_scheduler.experiments.jev_prompts import (
    ACTION_CRITERIA,
    BACKGROUND_EN,
    BACKGROUND_ZH,
    PROMPT_VERSION,
    PROMPTS,
)


def export(root: Path) -> None:
    """Export verbatim English prompts and Chinese translations without network calls."""
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Jev 实际提示词：英文发送，中文仅供阅读",
        "",
        f"版本：{PROMPT_VERSION}；由 scripts/export_jev_prompts.py 从实际代码常量生成。",
        "英文模板已做确定性注入接口测试；不是英文优于中文的效果证据或实网验证。",
        "",
        "## 完整执行顺序",
        "",
        "1. 生产计时内执行BILL4/P100初始化。",
        "2. 原strict/delta0门通过后，每一步LS开始前按当前解、真实细权重、轨迹反馈和累计动作历史重新问一次动作Choice。",
        "3. 每步只问当前子问题所属偏好组，由程序用本步新分布执行epsilon抽样，不跨步复用；初始化与非LS子代不发动作请求。",
        "4. 对全部合资格工件问Choice，程序抽样实际工件；再按所需询问phase、Region、path、position。",
        "5. 各层独立抽样，下一问绑定实际抽中对象。单选项直接使用，不额外调用；空集返回无候选。",
        "6. 一步最多六个不同完整候选，均从该步同一incumbent独立构造。统一exact与Archive、细权重局部接受。",
        "7. 第1至4步后用真实结果问Noul停止；大于0.5停，等于0.5继续，第5步强制结束。",
        "8. 原两重点Polish在轨迹后执行，随后原邻域替换。",
        "",
        "两版仅epsilon_g不同，分别0.5/0.9。动作、工件与条件位置均以该概率选最大项"
        "（并列均匀），否则按Jev完整分布抽样。这是概率引导epsilon-greedy变体，"
        "随机分支不是均匀探索。随机操作在程序中，模型不自行模拟。",
        "",
        "## 每次请求的共同背景",
        "",
        "英文（实际发送）：",
        "",
        *["> " + line for line in BACKGROUND_EN.splitlines()],
        "",
        "中文（不发送）：",
        "",
        *["> " + line for line in BACKGROUND_ZH.splitlines()],
        "",
    ]
    for key, (en, zh) in PROMPTS.items():
        lines += [f"## {key}", ""]
        for group in ("Flow", "Balanced", "Electricity") if key == "action" else ("",):
            e, z = (en.format(group=group), zh.format(group=group)) if group else (en, zh)
            lines += [
                f"英文（实际发送，{group or key}）：",
                "",
                "> " + e,
                "",
                "中文（仅阅读）：",
                "",
                "> " + z,
                "",
            ]
    lines += [
        "## 实际动态输入",
        "",
        "共同state含英文背景、真实拓扑/电价/资源、预算、曝光分母清晰的动作历史和Archive摘要。",
        "动作及对象请求都含真实细权重、完整MS/OS/时序、全部Job的P/D等待/资源/KV/指派、"
        "全部地区固定窗口功率。实际criteria由代码填充ID和英文结构化说明。"
        "A8 repair带partial_solution，不能把已移除成员当作仍固定。",
        "动作请求另含当前子问题索引、偏好组、归一化上下文、下一步编号和本轨迹已完成步。"
        "动作反馈累计自run开始，不因每步新回答而清空。request_mode为per_ls_step。",
        "名称不发送，只传稳定数值ID；未按评分截取工件。超过255选项按稳定ID区间分层，"
        "保留全部可达候选；区间描述含数量和首尾对象。分层不等价于全局一次Choice。",
        "",
        "## 算子权限",
        "",
        "| 算子 | Jev控制 | 程序保留 |",
        "| --- | --- | --- |",
        "| A1 | Job、phase、同地区实例 | Region和OS不变 |",
        "| A2 | Job、同地区P/D路径 | 合法容量和SSGS |",
        "| A3 | Job、目标地区、P/D路径 | P/D联合迁移，OS不变 |",
        "| A4 | Job、phase、全部合法前插位置 | 正等待、单工序前插 |",
        "| A5 | Job、P位置和条件D位置 | 正总等待、双前移、P在D前；不物化全部位置对 |",
        "| A6 | 无放回destroy工件集合 | 10%且至少2，原随机合法repair |",
        "| A7 | Job、phase、全部合法正压缩关键时刻 | 原关键时刻与非成员冻结 |",
        "| A8 | Region、singleton工件/phase或正边际support组、repair关键时刻 |"
        "前驱/窄窗顺序、成员/位置/2B尝试上限、原horizon、全窗削峰 |",
        "",
        "A7绕过原两代表/2B预筛，A8改变支撑组及位置选择，是研究策略，不冒称纯加速。"
        "A6 repair仍随机。A8部分位置只满足前驱边界，仍可能资源失败，按上限尝试，"
        "失败整recipe回滚。",
        "",
        "## 失败与预算",
        "",
        "不重试、不换模型。HTTP子进程硬截止kill/reap。动作请求失败或额度不足时，本步用均匀分布，"
        "不复用先前分布；每步记录action_source、action_request和action_query_s。"
        "对象失败在完整eligible集合均匀抽样并记录回退，"
        "分层回退按组大小加权。停止失败且连续两步无局部接受/Archive新目标时用本地规则停，"
        "否则继续，仍受L5和墙钟约束。",
        "默认每run最多512请求、单次15秒、累计等待最多生产预算50%、已计费或保守预留输入"
        "最多400万token。每次发送前按64k默认整请求上界预留，合法usage按实替换；未知失败"
        "保留整份预留。额度是单run限制，不是账号余额或多run全局总预算。",
        "输入、请求及等待都计入生产墙钟。当前未启动正式campaign或做新付费调用。",
        "",
        "## 两版运行",
        "",
        "模型固定jev-1.14.0。两版仅epsilon_g不同，配置见arms.json。密钥只从进程环境"
        "TYPESAFE_API_KEY读取，不写命令参数、prompt、日志或Git。",
        "",
        chr(96) * 3 + "powershell",
        "python -m geo_llm_scheduler.cli.jev --instance <original-component33-input.json>"
        " --epsilon-g 0.5 --seconds 300 --seed 18001 --output outputs/jev-e05",
        "python -m geo_llm_scheduler.cli.jev --instance <original-component33-input.json>"
        " --epsilon-g 0.9 --seconds 300 --seed 18001 --output outputs/jev-e09",
        chr(96) * 3,
        "",
        "时间/seed仅命令示例，正式需按原面板冻结。requests.jsonl保存脱敏完整请求/有效响应/"
        "失败；result.json保存最终P100/Archive、每个最终成员的独立exact审计和实际usage。",
        "实现与验收见IMPLEMENTATION.md。",
        "",
        "[官方Choice](https://docs.typesafe.ai/primitives/choice)；"
        "[官方Noul](https://docs.typesafe.ai/primitives/noul)；"
        "[Jev-Mem附录B](https://arxiv.org/html/2609.23986v1)。",
    ]
    (root / "PROMPTS_EN_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "arms.json").write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "action_request_mode": "per_ls_step",
                "arms": {
                    name: asdict(JevSettings(epsilon_g=eps))
                    for name, eps in (("JEV_E05", 0.5), ("JEV_E09", 0.9))
                },
                "action_criteria_english": ACTION_CRITERIA,
                "chinese_sent_to_provider": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    export(Path("research/jev_hierarchical_20261024"))
