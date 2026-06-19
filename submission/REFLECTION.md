# Reflection — Day 17 (≤ 200 words)

Answer briefly, in your own words. This is graded on reasoning, not length.

1. **The flywheel.** Day 13 emitted agent traces; today you turned them into an
   eval set and DPO pairs that Day 22 will train on. Which step in
   `traces → Bronze → datasets` would break most silently in production if you
   got it wrong — and how would you detect it?

2. **Decontamination.** Your run dropped 2 of 3 preference pairs because their
   prompts were in the eval set. What concretely goes wrong if you *skip* this
   step and train on those pairs? How would the lie show up in your metrics?

3. **Point-in-time.** The naive join leaked a future `lifetime_spend` into the
   training row. Describe one feature in a system you know that would be
   dangerous to join without an `ASOF`/point-in-time guard.

4. **Graph vs vector.** From `kg_demo.py`, name one question the knowledge graph
   answers well that flat chunk retrieval (`embed.py`) would struggle with, and
   one where the graph is overkill.

**1. Flywheel.** Bước **decontamination** dễ hỏng âm thầm nhất: lab chỉ khớp exact-match, nên prompt viết lại vẫn có thể rò vào train dù eval “sạch”. Phát hiện bằng audit overlap train/eval (n-gram hoặc embedding), so sánh eval offline với A/B production — metric eval cao nhưng traffic thật không cải thiện.

**2. Decontamination.** Bỏ qua bước này, model học thuộc đáp án đã có trong eval set (3 pair → 1 pair sạch). Eval win-rate/accuracy sẽ cao giả do leakage, không phải generalize; lie lộ ra khi test prompt mới hoặc deploy production vẫn kém dù benchmark “đẹp”.

**3. Point-in-time.** Feature **tổng chi tiêu tích lũy (lifetime spend)** — join “giá trị mới nhất” thay vì ASOF đưa spend tương lai vào dòng training, giống `naive_leaky_features` trong lab; offline tốt, inference thiếu dữ liệu chưa xảy ra nên sai.

**4. Graph vs vector.** KG trả lời tốt: *“Widget ship from đâu?”* — cần 2 hop (widget → accessory → Hanoi), không chunk nào chứa cả chuỗi. Graph thừa: *“Widget returnable không?”* — một fact trong một chunk, vector retrieval (`embed.py`) đủ.

---

## Extension — Compare decontamination (`compare_embed_models.py`)

Chạy: `python compare_embed_models.py` (Model A = exact, Model B = `gemini-3.1-flash-lite`, threshold = 0.55)

| | Model A (exact) | Model B (Gemini 3.1 Flash Lite) |
|---|---|---|
| Giữ lại | 2 pair | 1 pair |
| Loại | 2 | 3 |
| Paraphrase probe | sim **0.000** (bỏ sót) | sim **1.000** (bắt được) |

**Paraphrase test**
- Eval: *Can I return a widget I bought 10 days ago?*
- Pair: *is a widget bought ten days ago eligible for return?*

Exact-match chỉ loại prompt trùng chuỗi (lab: 3 raw → 1 sạch). Gemini loại thêm paraphrase cùng intent → Model B bắt **1 leak** mà Model A bỏ sót. Kết luận: decontamination exact-match không đủ production; cần semantic judge (LLM hoặc embedding) như đã nêu ở câu 1–2.
