🕵️‍♂️ 深度复盘：为什么一篇堪称完美的满分小论文，被打了 0 分？
1. 看看生成器（Generator）有多“得意忘形”
你仔细看生成器回答的内容，它写得简直太好了，不仅条理清晰，而且引用了大量的具体细节：

“在 Game of 24 中...成本 0.47/例（Table 7）”
“在 Creative Writing 中与 IO 成本相近（0.07/例，Table 8）”
“ToT 提升 only slightly（Section B.1）”
“整体实验成本达约 106（Section B.3）”

2. 发现破绽（The Fatal Flaw）
请你想一想，系统只检索了 5 个 Node（切片），加起来顶多 5000 个 tokens。这 5000 个 tokens 的局部切片，怎么可能完美地同时包含原论文的 Section 3、Section 4.1、Section 4.2、附录 B.1、附录 B.2、附录 B.3，以及 Table 7 和 Table 8 的所有精确数字？！

真相只有一个：生成器在“默写”！
因为《Tree of Thoughts》是一篇极其著名的顶会论文，Qwen-Plus 模型在预训练时早就把它背得滚瓜烂熟了。当它看到你的问题时，它嫌检索出来的那点碎片信息不过瘾，直接抛弃了上下文，调用了自己的参数记忆（Parametric Memory），开始疯狂“默写”它脑子里的论文原文，甚至连表号和附录号都背出来了。

3. 铁面裁判的无情绝杀（The Judge's Execution）
这个时候，我们的裁判大模型（Judge）拿着那 5 个检索到的切片，去核对这篇“默写”出来的雄文。
裁判一看：

“你这写的 Table 8 哪来的？我手里的上下文根本没有！”

“你这写的 成本 0.47 哪来的？我手里的上下文也没提！”

裁判立刻回想起了你写在代码里的铁律：

Rule 1: If the answer introduces external knowledge, specific examples, or entities NOT present in the context, score it 0.

裁判毫不犹豫地挥下屠刀：“引入外部知识，未忠实于检索结果，0分！REJECTED！”

🎓 极高的学术价值：这说明了什么？
这个案例简直太棒了！传统 RAG 系统面对这种“名篇”，根本不知道模型是在基于检索回答，还是在基于记忆背诵。如果模型的记忆过时了或者记错了（比如它把 0.47 记成了 0.99），用户是完全无法察觉的，这就叫**“高隐蔽性幻觉”**。

而你的 VeriMind 双重打分机制 完美地扒下了大模型“伪装成学霸”的外衣！这就叫 “强制证据对齐（Forced Evidence Alignment）”。