"""English search-clock contract; Chinese translations are documentation only."""

PROMPT_VERSION = "jev_single_v2_b1_a6_top10_search_clock"
CLOCK_EN = (
    "The production budget is SEARCH TIME: initialization, feature/prompt construction, "
    "local parsing/validation, candidate generation, exact evaluation, observers and "
    "search consume it. Synchronous API blocking (queue and HTTPS, including failed "
    "calls) is excluded and reported separately. API latency must not penalize action "
    "efficiency. History wall_s and derived rates use search time; completed-step "
    "seconds and wait_used_s are wall-time diagnostics. Remaining_s is remaining "
    "search time. Request/token limits still apply; no aggregate API-wait cutoff."
)
CLOCK_ZH = (
    "生产预算按搜索时间：初始化、特征和prompt构造、本地解析校验、候选构造、exact评价、"
    "观察日志及搜索均计时。同步API阻塞（排队、HTTPS，含失败请求）单独记录并扣除。"
    "不得因API延迟惩罚动作效率。历史wall_s及其收益率按搜索时间；已完成步seconds和"
    "wait_used_s仅为墙钟诊断。remaining_s是剩余搜索时间。请求/token上限仍有效，"
    "没有累计API等待截止门。"
)


def search_clock_text(text: str) -> str:
    """Adapt an English question's cost terminology without changing its options."""
    return text.replace("wall time", "search time").replace("wall cost", "search computation cost")
