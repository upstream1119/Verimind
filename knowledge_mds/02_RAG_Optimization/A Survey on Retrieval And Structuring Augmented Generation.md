# A Survey on Retrieval And Structuring Augmented Generation

> Auto-converted plain text. Category: 02_RAG_Optimization

A Survey on Retrieval And Structuring Augmented Generation
with Large Language Models
Pengcheng Jiang*
University of Illinois
Urbana-Champaign
Urbana, IL, USA
pj20@illinois.edu
Siru Ouyang*
University of Illinois
Urbana-Champaign
Urbana, IL, USA
siruo2@illinois.edu
Yizhu Jiao
University of Illinois
Urbana-Champaign
Urbana, IL, USA
yizhuj2@illinois.edu
Ming Zhong
University of Illinois
Urbana-Champaign
Urbana, IL, USA
mingz5@illinois.edu
Runchu Tian
University of Illinois
Urbana-Champaign
Urbana, IL, USA
runchut2@illinois.edu
Jiawei Han
University of Illinois
Urbana-Champaign
Urbana, IL, USA
hanj@illinois.edu
Abstract
Large Language Models (LLMs) have revolutionized natural lan-
guage processing with their remarkable capabilities in text genera-
tion and reasoning. However, these models face critical challenges
when deployed in real-world applications, including hallucination
generation, outdated knowledge, and limited domain expertise. Re-
trieval And Structuring (RAS) Augmented Generation addresses
these limitations by integrating dynamic information retrieval with
structured knowledge representations. This survey (1) examines re-
trieval mechanisms including sparse, dense, and hybrid approaches
for accessing external knowledge; (2) explore text structuring tech-
niques such as taxonomy construction, hierarchical classification,
and information extraction that transform unstructured text into
organized representations; and (3) investigate how these struc-
tured representations integrate with LLMs through prompt-based
methods, reasoning frameworks, and knowledge embedding tech-
niques. It also identifies technical challenges in retrieval efficiency,
structure quality, and knowledge integration, while highlighting
research opportunities in multimodal retrieval, cross-lingual struc-
tures, and interactive systems. This comprehensive overview pro-
vides researchers and practitioners with insights into RAS methods,
applications, and future directions.
CCS Concepts
• Information systems →Information retrieval; • Computing
methodologies →Natural language processing; Knowledge
representation and reasoning.
Keywords
Text Mining, Large Language Models, Retrieval Augmented Gener-
ation
* P. Jiang and S. Ouyang contributed equally.
This work is licensed under a Creative Commons Attribution 4.0 International License.
KDD ’25, Toronto, ON, Canada
© 2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1454-2/2025/08
https://doi.org/10.1145/3711896.3736557
ACM Reference Format:
Pengcheng Jiang*, Siru Ouyang*, Yizhu Jiao, Ming Zhong, Runchu Tian,
and Jiawei Han. 2025. A Survey on Retrieval And Structuring Augmented
Generation with Large Language Models. In Proceedings of the 31st ACM
SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’25),
August 3–7, 2025, Toronto, ON, Canada. ACM, New York, NY, USA, 15 pages.
https://doi.org/10.1145/3711896.3736557
1
Introduction
Large Language Models (LLMs) have revolutionized natural lan-
guage processing, demonstrating unprecedented capabilities in
tasks ranging from text generation to complex reasoning [2, 17].
These models, trained on vast amounts of text data, have shown re-
markable abilities in understanding context, generating human-like
responses, and adapting to various tasks with minimal instruc-
tion [190, 258]. However, when deployed in real-world applications,
LLMs face several critical challenges: they can generate plausible
but factually incorrect information (hallucination) [82, 86], rely
on potentially outdated training data [124, 171], and show limited
expertise in specialized domains [195, 227].
The limitations of LLMs become particularly apparent in knowledge-
intensive applications where accuracy and reliability are paramount.
While these models excel at pattern recognition and text generation,
they struggle with maintaining factual consistency and accessing
current information. In specialized fields such as scientific research,
healthcare, or technical domains, LLMs often lack the precise and
detailed knowledge required for reliable performance [50, 95, 145].
To address these challenges, Retrieval-Augmented Generation
(RAG) was proposed as a framework that enhances LLM capabilities
by retrieving relevant information from external knowledge sources
before generating responses [64, 124]. By grounding LLM responses
in retrieved documents, RAG reduces hallucinations and enables
access to up-to-date information not present in the model’s pa-
rameters. Despite these advancements, traditional RAG approaches
face fundamental limitations: they typically process unstructured
text passages without leveraging structured knowledge, contain
non-atomic information that can mislead LLMs [50, 95, 261], and
struggle with complex queries requiring multi-hop reasoning or
domain-specific knowledge organization [303, 306].
arXiv:2509.10697v1  [cs.CL]  12 Sep 2025


KDD ’25, August 3–7, 2025, Toronto, ON, Canada
Pengcheng Jiang et al.
Retrieval And Structuring (RAS) has emerged as a more powerful
paradigm that addresses these limitations by integrating dynamic
information retrieval with structured knowledge representations.
While RAG provides the foundation for connecting LLMs with
external information, RAS extends this capability by incorporating
knowledge structuring techniques that transform unstructured text
into organized representations such as taxonomies, hierarchies, and
knowledge graphs [46, 224, 299]. These structured representations
help organize retrieved information, guide the retrieval process,
and provide a framework for verifying LLM outputs.
Recent advances in RAS have demonstrated significant improve-
ments in LLM performance through novel retrieval strategies that
better align with LLM reasoning patterns [63, 252] and improved
methods for constructing high-quality knowledge representations [90,
116]. The integration of retrieval and structuring mechanisms has
opened new possibilities across domains: in scientific research,
RAS systems synthesize information while maintaining technical
accuracy [37, 275]; in e-commerce, they generate personalized rec-
ommendations [26, 279]; and in healthcare, they provide reliable
contextualized information [110, 259].
Several challenges remain in fully realizing RAS potential: effi-
cient retrieval at scale [238, 314], maintaining high-quality knowl-
edge representations [306], seamlessly integrating structured in-
formation with LLM reasoning [151, 230], and balancing computa-
tional overhead with real-time performance [48, 128].
This survey provides a comprehensive examination of RAS, its
components, applications, and future directions. We begin with
preliminaries on LLMs and retrieval-augmented generation, then
examine advanced retrieval mechanisms and text structuring ap-
proaches, including taxonomy construction, hierarchical classifica-
tion, and information extraction. We analyze how these structured
representations integrate with LLMs to improve reasoning and
adaptation capabilities, examine domain-specific applications, and
conclude with technical challenges and research opportunities.
2
Preliminaries and Foundations
This section provides a concise overview of the preliminaries and
foundational aspects of Large Language Models (LLMs) and Retrieval-
Augmented Generation (RAG).
2.1
Large Language Models
2.1.1
Architectures. LLMs are primarily built upon three architec-
tural frameworks: (1) Encoder-only Models, like BERT [44] and
its derivatives, designed for understanding tasks, process input text
to generate embeddings or predict class labels, excelling in appli-
cations such as text classification and information extraction. (2)
Encoder-Decoder Models, like BART [123] and T5 [206], consist
of both encoder and decoder components, enabling them to trans-
form input text into output text. This architecture is particularly
effective for natural language generation tasks like machine trans-
lation and text summarization. (3) Decoder-only Models, like
GPT [2] and DeepSeek [42], are optimized for generative tasks. By
predicting subsequent tokens in a sequence, these models excel in
open-ended applications like text completion and dialogue genera-
tion, establishing their dominance in the generative AI landscape.
2.1.2
Training Paradigms. The development of LLMs involves sev-
eral key stages: (1) Pre-training. Initially, models are exposed to
extensive text corpora to learn language patterns through tasks
like Masked Language Modeling [44] and Autoregressive Language
Modeling [204], building a foundational understanding of linguistic
structures. (2) Supervised Fine-Tuning (SFT). After pre-training,
LLMs can undergo instruction tuning to better follow human in-
structions rather than merely predicting the next tokens [191].
This process trains models on instructional prompts and responses,
enhancing their ability to interpret and execute user requests. Addi-
tionally, models can be fine-tuned either from a pre-trained state or
an instruction-tuned version to optimize performance for specific
tasks. (3) Reinforcement Learning (RL). Reinforcement learn-
ing techniques refine model behavior by aligning outputs with
human preferences using methods like Proximal Policy Optimiza-
tion (PPO) [217] and Direct Preference Optimization (DPO) [205].
Although RL typically follows SFT stage, it can also be applied
directly to pre-trained models, as shown in DeepSeek-R1-Zero [43].
2.1.3
Inference Strategies. Modern LLMs leverage advanced in-
ference techniques to enhance performance and scalability: (1)
In-Context Learning (ICL). ICL [18] enables models to adapt to
new tasks by incorporating examples directly into the input prompt,
facilitating task-specific responses without additional training. (2)
Chain-of-Thought (CoT). CoT prompting [258] guides models to
generate intermediate reasoning steps, boosting performance on
complex tasks by making the problem-solving process explicit. (3)
Test-Time Scaling. Recent work, such as OpenAI’s o1 [189], has
explored strategies to improve reasoning capabilities by allocating
more computational resources during inference. This approach en-
ables models to perform more extensive deliberation, like longer
CoT reasoning, before generating responses. By scaling computa-
tion effort at test time, models can achieve substantial improve-
ments on complex tasks, particularly in science and mathematics.
2.2
Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) enhances LLMs by retriev-
ing relevant information from external sources before generating
responses [124]. RAG addresses hallucination, outdated knowledge,
and domain expertise limitations by grounding responses in re-
trieved documents [82, 86]. The RAG pipeline includes indexing
(document processing/vectorization), retrieval (identifying relevant
information), and generation (with retrieved context) [16, 64].
2.2.1
RAG Paradigms. (1) Naive RAG. This initial approach em-
ploys a straightforward "retrieve-read" framework [159], where
documents are retrieved based on the original query and directly
fed into the model. While effective, this approach struggles with
retrieval precision and may introduce irrelevant information that
can mislead LLMs [24, 95, 261]. (2) Advanced RAG. This para-
digm introduces pre-retrieval and post-retrieval optimizations. Pre-
retrieval techniques focus on improving indexing structures and
query formulation through query rewriting [159], query expansion
[198], and hybrid retrieval approaches [63]. Post-retrieval processes
enhance integration through reranking and context compression
techniques [14], significantly improving response relevance. (3)


A Survey on Retrieval And Structuring Augmented Generation with Large Language Models
KDD ’25, August 3–7, 2025, Toronto, ON, Canada
Modular RAG. The most recent paradigm represents a flexible ar-
chitecture incorporating specialized components for specific tasks
[251]. This approach introduces functional modules like search in-
terfaces, memory systems, and task adapters, enabling sophisticated
retrieval strategies [114] and patterns like rewrite-retrieve-read
[159], generate-read [281], and demonstrate-search-predict [112].
2.2.2
Key Components and Techniques. (1) Retrieval Sources.
While unstructured text data remains predominant [249], research
explores semi-structured data [133], structured knowledge graphs
[106, 251], and even LLM-generated content [30, 234]. (2) Retrieval
Granularity and Embeddings. Granularity ranges from fine-
grained (tokens, sentences) to coarse-grained (chunks, documents)
units [28], each balancing information completeness against re-
trieval precision. Embedding models have seen substantial improve-
ments through specialized architectures [135, 185] and alignment
fine-tuning [40, 291]. (3) Context Processing. Reranking meth-
ods [315] prioritize the most relevant documents while context
compression techniques [267] reduce redundancy while preserving
essential content, addressing the "lost in the middle" problem [144].
2.2.3
Advanced Workflows and Integration. RAG has evolved be-
yond static pipelines to include adaptive retrieval [6, 98], where
models autonomously determine when retrieval is necessary; it-
erative retrieval [221], which employs recursive cycles to refine
responses; and recursive retrieval [114], which hierarchically de-
composes complex queries. Despite increasing context windows
in modern LLMs, RAG maintains importance through efficiency,
transparency, and domain adaptability [194, 262]. The approach
continues to evolve through integration with fine-tuning [140, 154],
creating hybrid systems. Recent evaluations reveal that while fine-
tuning shows some improvement, RAG consistently outperforms
it for both existing and new knowledge [193]. Research has also
extended RAG to incorporate other modalities like images [274]
and code [136].
3
Information Retrieval
Information Retrieval (IR) [52, 70, 174] aims to identify and retrieve
relevant documents from a corpus based on queries. It plays a crucial
role in various tasks including internet browsing [115, 122, 201],
question-answering systems [1, 111, 238], and LLM applications [53,
64, 303]. In the RAS paradigm, IR serves as a critical component
that provides external, up-to-date, yet unstructured knowledge. The
quality of the IR process directly impacts the overall effectiveness
of the whole framework.
3.1
Classical Retrieval Methods
3.1.1
Sparse Retrieval. Sparse retrieval has long been the de facto
standard for information retrieval [67, 70, 150]. At the core, it repre-
sents text using normalized term frequency and inverse document
frequency.
Fundamental Approaches. A fundamental approach in this cate-
gory is TF-IDF [229], which scores terms based on their frequency
in a document while penalizing common words that appear across
many documents. BM25 [210, 211], an improvement over TF-IDF,
refines this approach by introducing a saturation function for term
frequency and document length normalization.
Statistical Modifications. Several statistical variations of BM25
have been introduced [240]. BM25F [199, 210, 226] extends BM25
for fielded search by considering term weights across multiple docu-
ment fields. BM25+ [156] introduces different frequency saturation
mechanisms. Other variants include BM25-adpt [155], LM-DS [284],
and LM-PYP [175], each optimizing the BM25 formula for different
retrieval scenarios.
Neural Modifications. Learned sparse retrieval incorporates neu-
ral networks to generate sparse, term-based representations [39, 58,
139, 161]. DeepCT [39] and uniCOIL [139] leverage transformers to
re-weight terms, assigning weights based on semantic importance.
SPLADE [58] utilizes masked language modeling to predict addi-
tional relevant terms, blending semantic strengths of deep learning
with efficiency of inverted indexes.
Impact. Even in the deep learning era, sparse retrieval remains a
strong baseline due to its effectiveness, efficiency and interpretabil-
ity. It continues to be competitive with more complex neural mod-
els, particularly with limited training data or domain-specific con-
straints [238]. However, term-based scoring cannot retrieve relevant
content without lexical overlap and can miss broader context or
meaning [4, 150].
3.1.2
Dense Retrieval. In contrast to sparse lexical methods, dense
retrieval [61, 304, 314] uses continuous vector embeddings to rep-
resent queries and documents. With pre-trained language mod-
els [44, 180, 206], relevant query–document pairs have high cosine
similarity, enabling semantic matching without lexical overlap.
Dual Encoder and Early Dense Retrieval. Early dense retrieval
emerged as a response to the limitations of sparse retrieval. Karpukhin
et al. [111] designed Dense Passage Retrieval (DPR), which showed
that a BERT [44]-based dual encoder trained on question–answer
pairs [10, 11, 104, 119, 207] can surpass BM25 [210, 211] in open-
domain question answering. Early models struggled with gen-
eralization, requiring large-scale labeled datasets [179] and ap-
propriate negative sampling strategies [266] in contrastive learn-
ing [27, 51, 75, 188].
Effectively-trained Dense Retrieval. Recognizing limitations of
static negative sampling, Asynchronous Negative Contrastive Esti-
mation (ANCE) [266] addressed this by refreshing hard negatives
during training. RocketQA [203] improved contrastive training
with cross-batch memory for more diverse negative samples. Gen-
eralizable T5 Retriever (GTR) [181] framed retrieval as a sequence-
to-sequence task rather than solely relying on contrastive learning.
Efficient and Scalable Dense Retrieval. Another research line
focused on scalability and efficiency. ColBERT [113] introduced
late interaction mechanisms, preserving token-level representa-
tions to reduce memory overhead. CoCondenser [62] optimized
intermediate representation learning for more compact document
embeddings. E5 [246] explored improved contrastive pretraining
techniques for adaptability to various tasks.
Impact. Dense Retrieval models [111, 113, 181, 246, 266] offer sig-
nificant advantages over Sparse Retrieval by enabling semantic
matching when exact lexical overlap is absent. However, Dense
Retrieval is limited by dependency on large-scale labeled training
data [54, 131, 304] and lack of interpretability [31, 105, 148].


KDD ’25, August 3–7, 2025, Toronto, ON, Canada
Pengcheng Jiang et al.
3.1.3
Others. In addition to sparse and dense retrieval, alternative
methods exist for retrieving relevant text.
Retrieval with Edit Distance. Edit distance-based Retrieval matches
queries and documents by comparing edit distances between words
and phrases like Levenshtein Distance [69, 77, 282]. These methods
are especially popular in code retrieval [74], often integrated with
abstract syntax trees of code snippets [200, 287].
Retrieval with Named Entity Recognition. Named Entity Recog-
nition (NER) [130, 177, 270]-based retrieval uses entities in queries
and documents for matching, allowing text to be understood at
a concept level [141, 268]. Entity-based retrieval enforces type
matching, especially useful for specific scenarios like legal do-
main [49, 121].
Retrieval with Large Language Models. Recently, Generative
Retrieval (GR) [13, 41, 231, 236, 252] has emerged as a new par-
adigm leveraging generative model [2, 206, 239] parameters for
memorizing documents, enabling retrieval by directly generating
relevant document identifiers. Given the same parameter size and
training data, GR can achieve better performance [236, 252] and
interpretability [236, 290], becoming a new research focus [134].
3.2
Recent Retrieval Strategies
While building on major retrieval frameworks discussed above,
diverse strategies have been recently introduced at various stages
of the retrieval pipeline.
3.2.1
Hybrid Retrieval. Hybrid retrieval [63, 150, 162] combines dif-
ferent retrieval models to leverage complementary strengths. Gao
et al. [63] proposed CLEAR (Complementing Lexical Retrieval with
Semantic Residual), learning a dense "residual" vector to comple-
ment BM25. Luan et al. [150] explicitly combined sparse and dense
representations, achieving better ranking accuracy than strong
single-method baselines. This fused retrieval fashion enhances ef-
fectiveness and robustness [241, 286].
3.2.2
Data Augmentation for Retrieval. To enlarge training scale,
data augmentation [15, 21, 183, 184] leverages additional data be-
yond original labeled queries and documents. Doc2query [184]
trained a transformer [242] to predict likely queries for each docu-
ment. DocT5query [183] used T5 [206] to generate multiple queries
per document. Inpars [15] used GPT-3 [17] to generate queries for
documents in target domains. These techniques inject additional
knowledge into retrieval systems, becoming key drivers of perfor-
mance improvement.
3.2.3
Query Expansion and Rewriting. Query expansion [7, 19, 178,
213] and rewriting [143, 159, 277] aim to improve retrieval by refor-
mulating the input query. Roy et al. [213] expanded queries with
semantically related terms using word2vec [172]; Naseri et al. [178]
applied BERT to contextualize expansion based on retrieved docu-
ments. Ma et al. [159] trained LMs to rewrite queries using initial
retrieval outputs. Jiang et al. [92] introduced DeepRetrieval, an RL-
based approach that optimizes retrieval metrics (e.g., Recall@K,
NDCG) directly without supervision, and achieves SOTA perfor-
mance across literature, evidence-seeking, and SQL retrieval tasks
with only 3B parameters. Experimental results show that effective
query rewriting/expansion significantly improves retrieval quality.
Building on this line, s3 [96] further demonstrates that training a
search-only agent with reinforcement learning—using a generation-
aware reward rather than search-only metrics—can yield strong
downstream improvements while requiring orders of magnitude
less training data.
3.2.4
Multi-staged Retrieval and Reranking. For efficiency and effec-
tiveness, reranking processes [27, 35, 60, 118, 182] adopt two-stage
retrieval strategies, where documents from a fast method are re-
scored by a more powerful but slower ranking model. Nogueira
and Cho [182] demonstrated BERT-based rerankers dramatically
improve ranking quality. As NLP advances, LLMs [17, 206, 239]
have emerged as state-of-the-art rerankers, boosting retrieval per-
formance by large margins on both general [27, 35, 60, 118] and
specific domains [3].
4
Text Structuring
Text structuring is essential for converting unstructured text into or-
ganized, interpretable knowledge for analysis and decision-making.
This section explores three key text mining methodologies: tax-
onomy construction and enrichment, text classification, and infor-
mation extraction. These approaches effectively organize textual
information, enabling downstream tasks like knowledge graph con-
struction, database population, and tabular relational learning.
4.1
Text Mining
4.1.1
Taxonomy. Taxonomy is a tree-like structure that organizes
concepts into a hierarchical framework, where parent nodes rep-
resent broader categories, and child nodes capture finer-grained
subcategories. Taxonomy is often used as the label space structuring
method for text-mining tasks.
Taxonomy Construction. Taxonomy construction typically be-
gins with a seed structure. HiExpan [224] expands entity taxonomies
through width and depth expansions, followed by global structure
adjustment using word analogy [202]. CoRel [81] leverages pre-
trained language models (PLMs) to train a relation-transferring
module that generalizes seed parent-child relationships. TaxoCom [120]
completes partial topical taxonomies by learning local discrimina-
tive word embeddings and applying novelty-adaptive clustering.
A related task is set expansion, which extends a seed set with
semantically similar entities. SetExpan [223] uses skip-grams and
word embeddings to measure similarity between candidates and
the seed set. Set-CoExpan [80] introduces auxiliary sets through
embedding learning. CGExpan [298] exploits PLMs to generate
knowledge-probing queries, while ProbExpan [137] refines entity
representations using contrastive learning. FGExpan [263] focuses
on fine-grained set expansion, targeting specific common types.
Taxonomy Enrichment. Taxonomy enrichment enhances each
taxonomy node with discriminative texts, such as keywords. It
can be classified into flat and hierarchical enrichment. Flat enrich-
ment methods often use PLM embeddings for clustering [169, 302]
or discriminative topic modeling. CatE [164] creates a joint word
embedding space, while KeyETM [72] extends embedded topic
modeling [45] by integrating topical keywords. GTM [36] focuses
on guided topic modeling for short text. SeeTopic [297] addresses


A Survey on Retrieval And Structuring Augmented Generation with Large Language Models
KDD ’25, August 3–7, 2025, Toronto, ON, Canada
unseen topic names by leveraging PLMs’ general knowledge. Seed-
TopicMine [300] explores diverse context information for topic mod-
eling. Hierarchical enrichment requires modeling additional struc-
ture information. TaxoGen [285] recursively clusters word embed-
dings and refines them using local corpora. NetTaxo [219] extends
this by incorporating network structure information. JoSH [170]
leverages a taxonomy to train a joint embedding space that pre-
serves relative tree distances between nodes.
4.1.2
Text Classification. Text classification categorizes text into
different categories. According to how the label space is organized,
it can be divided into flat classification (independent categories)
and hierarchical classification (tree-structured labels).
Flat Classification. Flat classification has been extensively studied.
Early works [272, 294] focused on designing task-specific objectives
for different architectures. Later, external supervision sources such
as knowledge bases [228], human-curated rules [8, 20, 208], and
keywords [163, 165, 187, 209] were utilized. Keywords inspired
self-training methods like WeSTClass [165], LOTClass [168], and
X-Class [255]. ClassKG [289] combined keywords and knowledge
bases by constructing a keyword graph using GNNs.
With the rise of LLMs, prompting-based methods [71, 245, 260,
313] achieved notable results by leveraging LLMs’ internal knowl-
edge. Min et al. [173] proposed channel models for few-shot learn-
ing, estimating conditional probabilities. NPPrompt [305] intro-
duced a zero-shot approach that builds class verbalizers using PLM
embeddings. PIEClass [296] explored discriminative PLM prompt-
ing for pseudo-labeling. CARP [232] leveraged LLMs’ reasoning
abilities for zero and few-shot classification, using CoT prompting
to identify indicative clues before labeling.
Hierarchical Classification. Hierarchical text classification as-
signs input text to one or more nodes within a taxonomy-structured
label space, presenting a greater challenge than flat classification
due to its extensive structure. Most studies adopt either fully-supervised
or semi-supervised approaches. Banerjee et al. [9] and Wehrmann
et al. [257] train multiple classifiers for each node or level of the
taxonomy. Others [23, 97, 196, 254, 256] train a unified classifier for
the entire taxonomy. However, these methods heavily rely on man-
ual annotation, making weakly-supervised approaches increasingly
popular. WeSHClass [166] extends WeSTClass [165] by modeling
label hierarchies as mixtures of distributions and using self-training.
HiMeCat [295] integrates metadata by learning joint representa-
tions for labels, metadata, and text using a hierarchical generative
model, which supports weak supervision through augmented doc-
uments. TaxoClass [222] leverages a pre-trained entailment model
and employs taxonomy-guided search for pseudo-training samples.
TELEClass [299] enhances taxonomies with class-indicative fea-
tures, adapting LLMs for hierarchical classification by selecting
pseudo labels from retrieved candidates and generating pseudo
documents for long-tail and fine-grained classes.
4.1.3
Information Extraction. Information extraction is another
important task for text mining, focusing on entity-level information
as building blocks for knowledge structuring.
Entity Mining. Entity mining includes named entity recognition
(NER) and fine-grained entity typing (FET). NER involves extracting
entities from text before recognition. Early approaches addressed
uncertainty in distantly supervised labels: AutoNER’s "Tie or Break"
strategy [218], PaTTER’s dictionary expansion [250], and ETAL’s
confident entity identification [22]. RoSTER [167] addressed incom-
plete labels using noise-robust learning. SDNET [25] pre-trained on
Wikipedia entities. SEE-Few [273] utilized an entailment framework
for learning from limited examples. Recent LLM approaches include
GPT-NER [247] with self-verification, Shang et al. [220]’s document
graphs, and X-NER [197] for extremely weakly supervised settings.
FET deals with a more detailed label space organized into an on-
tology. ZOE [307] utilized a taxonomy based on Freebase types.
OntoType [116] aligned fine-grained types with an ontology using
weak supervision. OnEFET [192] enriched ontology structures with
instances. ALIGNIE [79] employed prompt-based approaches for
interpreting entity types. SEType [301] incorporated additional en-
tities from unlabeled corpora. Ultra-fine entity typing (UFET) [34]
expanded the label space to thousands of types. BERT-MLMET[38]
fine-tuned BERT with supervision from headwords and hypernyms.
LITE [126] leveraged natural language inference for entity typing.
DenoiseFET [132] clustered the extensive label space into centroids
for better generalization.
Relation Extraction. Building on mined entities, relation extrac-
tion (RE) identifies relationships between entity pairs, playing
a crucial role in knowledge graph construction (Sect. 4.2) and
event induction [101, 109, 225]. Leveraging LLMs, Wadhwa et al.
[243] demonstrate that few-shot prompting in GPT-3 can rival
state-of-the-art fully supervised RE models and explore LLM-based
data augmentation with CoT explanations. Wan et al. [244] en-
hance in-context learning by utilizing task-aware sentence-level
and entity-level representations for kNN retrieval of examples. Re-
lationPrompt [32] prompts LLMs to synthesize relation examples
for generating structured texts. To minimize hallucinations and
enhance factual accuracy, STAR [157] and DocGNRE [129] adopt
self-refinement and self-validation strategies. REPaL [309] suggests
that relation descriptions offer broader semantic coverage than
mere relation names. However, LLMs show limitations in relational
reasoning, possibly due to the scarcity of RE tasks in instruction-
tuning datasets [253, 288]. Addressing this, Ma et al. [160] pro-
pose the LLM-SLM cooperation framework, a filter-then-rerank ap-
proach, showcasing LLMs’ effectiveness in information extraction.
QA4RE [288] reframes RE as a question-answering task to align
with instruction-tuning, while SumAsk [127] combines summariza-
tion and QA. GenRES [93] criticizes traditional evaluation metrics
like precision and recall for generative RE and introduces a com-
prehensive multi-dimensional evaluation framework. Additionally,
Obamuyide and Vlachos [186] and Sainz et al. [214] reformulate RE
as a natural language inference (NLI) task. REBEL [83] redefines re-
lation triplet extraction as a Seq2seq task using decoder-based LMs,
while SURE [149] adopts a summarization approach by verbalizing
relation names. Prompt-tuning models [29, 71] also improve rela-
tional reasoning by integrating diverse prompts. Another research
direction focuses on aggregating relation-indicative evidence. RE-
PEL [202] co-trains a pattern-based module with a distributional
module for RE, while RClus [308] leverages patterns of relation-
specific entity types and indicative words in dependency paths.
Eider [265] adopts a document-level approach to extract sentence-
level evidence, and SAIS [264] supervises intermediate steps like


KDD ’25, August 3–7, 2025, Toronto, ON, Canada
Pengcheng Jiang et al.
coreference resolution and entity typing to retrieve fine-grained,
relation-specific evidence.
4.2
Knowledge Structuring
Structured knowledge, such as knowledge graphs, databases, and
tabular data, has long been studied in knowledge structuring [306].
Knowledge Graph Construction. Knowledge graphs (KGs) rep-
resent real-world entities as nodes and their relationships as edges.
In the general domain, open information extraction (OpenIE) tools
extract triples from raw documents, which are then filtered, linked,
and merged to form KGs. However, pipelined methods often suffer
from error propagation, leading to the development of end-to-end
approaches. For instance, Zeng et al. [283] propose a sequence-
to-sequence model that jointly extracts entities and relations, ad-
dressing the overlap problem through a copy mechanism. Similarly,
frameworks like REBEL [83] and ABSA [271] leverage pre-trained
language models (PLMs) to translate raw text into structured knowl-
edge. TagReal [89] uses pattern mining to generate high-quality
prompts for KG completion. With the rise of large language models
(LLMs), new methods for KG construction have emerged. Knowl-
edgeGraph GPT [235] directly prompts GPT-4 to convert plain text
into KGs. AutoKG [312] employs multiple LLM agents to enhance
KG construction and reasoning. KG-FIT [90] refines KG embed-
dings by constructing coherent entity hierarchies and fine-tuning
them using both hierarchical and textual information. Domain-
specific KG construction requires substantial human input for anno-
tations and ontology development. For example, Rotmensch et al.
[212] integrates annotated medical records with existing KGs to
build a health-related KG. PAIR [59] uses LLMs as relation filters to
streamline relation discovery in online marketing. In the biomedical
field, Karim et al. [110] build ontologies to validate gene-disease
relationships and use BioBERT to generate RDF triples, later re-
fined by LLMs. MedKG [91] emphasizes human-in-the-loop interac-
tion by allowing users to modify nodes and edges. In e-commerce,
FolkScope [280] adopts a semi-automated approach using domain-
specific prompts to guide LLMs, while Chen et al. [26] investigates
prompting strategies for effective KG construction in few-shot set-
tings. Theme-specific KG construction is more challenging for its
focus on specialized or fast-evolving domains. TKGCon [47] ad-
dresses this by starting with a theme-specific entity ontology from
Wikipedia and using LLMs to generate relation ontologies. Entities
and relations are mapped to the corpus to form the final KG.
Database Population. Databases are structured systems with ex-
ecutable operations to manage data [103]. AVATAR [117] estab-
lished probabilistic database techniques for IE. Li et al. [128] ex-
amines LLMs as database interfaces. DB-GPT [311] proposes an
LLM-based framework for database-specific tasks. D-Bot [310] au-
tomatically diagnoses databases. ChatDB [78] augments LLMs with
SQL databases for complex reasoning. Text2DB [278] designs an
agent system to automatically populate databases from text.
Tabular Data Organization. Tables are a prevalent structured
format for various applications. RolePred [99] utilizes information
redundancy in multiple documents to extract structured tables
including the attribute names and the corresponding values of an
event type. ODIE [100] adopts instruction tuning for large language
models to extract a single table from the texts following on-demand
Taxonomy
Retriever
LLM
KG
Thematically & Semantically 
Relevant Text
Retrieval
Structuring
Subgraph
Feedback & 
Subquery
Update
Query
Generation
Figure 1: An abstractive example of RAS paradigm.
user instructions. In particular, the table structures can be defined by
users or automatically inferred by the model. TableLLAMA [292]
and Table-GPT [133] also fine-tune large language models with
table-related data to improve their ability to process tables, like row
population, column addition, and cell filling. StructGPT [87] further
design specialized interfaces to interact with tables for reasoning
tasks. EVAPORATE [5] transforms semi-structured documents into
tabular views. RelBench [57] reformulates relational databases as
heterogeneous graphs for representation learning with GNNs.
5
Retrieval-And-Structuring
The Retrieval-And-Structuring (RAS) paradigm unifies three com-
plementary approaches to knowledge processing: information re-
trieval, structured knowledge representation through Knowledge
Graphs (KGs), and Large Language Models (LLMs). This integrated
framework enhances LLMs’ ability to perform knowledge-intensive
tasks, enables more effective knowledge discovery within struc-
tured knowledge, and creates synergistic benefits across all three
components. In this section, we examine the foundational princi-
ples of this paradigm, analyze recent methodological advances, and
explore promising directions for future research.
5.1
Structure-Enhanced Retrieval
Taxonomy-based approaches. Classic retrieval methods [58, 84,
111, 113, 210, 215], while effective for general search, are not op-
timized for theme-specific searches with explicit thematic focus.
In contrast, taxonomy-guided retrieval utilizes the thematic struc-
tures captured by the domain-specific topic taxonomy structures,
which does not only align closely with the topics and subtopics
in a taxonomy but also bypasses the requirement for large-scale
labeled/annotated data. Along this track, ToTER [107] augments
dense retrieval by leveraging corpus taxonomies through three
strategies: search space filtering via topic overlap, relevance match-
ing through topic distributions, and query enrichment using taxonomy-
guided phrases, which allows improved domain-specific search
without requiring labeled data. TaxoIndex [108] extracts key con-
cepts from papers and organizes them as a semantic index guided by
academic taxonomies at both topic and phrase levels, then trains an
add-on module to identify these concepts in queries and documents
to improve matching of underlying academic concepts in dense
retrievers, achieving strong performance even with limited training
data. In a similar vein, Lima et al. [237] demonstrates that retrieval


A Survey on Retrieval And Structuring Augmented Generation with Large Language Models
KDD ’25, August 3–7, 2025, Toronto, ON, Canada
performance varies significantly across different query taxonomy
classes, reinforcing the importance of structured categorization for
optimizing retrieval.
KG-based approaches. Knowledge graph, as another structured
data, also demonstrated its effectiveness to improve the retrieval
quality. For instance, KG-RAG [216] reduces LLM hallucinations [86]
by constructing KGs from text and retrieving information through
a Chain of Explorations algorithm, which sequentially traverses the
graph using LLM reasoning to find relevant answers with higher
precision than vector-based retrieval. Xu et al. [269] leverage KGs
for retrieval by parsing issue tickets into tree structures with in-
terconnected relationships, then using entity matching and graph
traversal to locate relevant information instead of relying on text-
chunk similarity search. HippoRAG [68] constructs a KG from
passages using LLM-based triple extraction, then leverages person-
alized PageRank [73] to explore graph neighborhoods around query
entities to identify relevant passages with implicit connections. Sim-
ilarly, GFM-RAG [153] follows this KG construction approach but
innovates in the retrieval mechanism by employing a pre-trained
query-dependent graph foundation model, enabling more sophisti-
cated multi-hop relationship reasoning in a single step.
Inspired by the recent advancements in retrieval strategies in-
troduced in Section 3, we depict a promising RAS method in Fig-
ure 1. The method combines taxonomy and KG to enable precise
knowledge retrieval. The process follows an iterative cycle where
a taxonomy-enhanced retriever first identifies both thematically
and semantically relevant documents from the corpus. These doc-
uments are structured into a subgraph, which updates a query-
specific evolving KG. The KG augments the LLM’s capabilities by
providing essential, structured information for answering the query.
If additional information is needed, the LLM generates a focused
subquery conditioned on the KG, initiating another retrieval cycle
until sufficient knowledge is gathered. This feedback loop ensures
comprehensive and accurate responses by dynamically building
and refining a knowledge context tailored to the query.
5.2
Structure-Enhanced LLM Generation.
Once knowledge is structured into a comprehensive KG, the chal-
lenge shifts to leveraging this representation to enhance LLM out-
puts. Structure-enhanced LLM generation grounds model responses
in explicit knowledge structures, reducing hallucinations and im-
proving factual consistency.
5.2.1
Early Approaches (PLM + KG). Early integration approaches
broadly fall into neural and rule-based categories: (1) Neural In-
tegration. Initial neural methods fused KGs with pre-trained lan-
guage models through various architectural innovations. KG-BART
[147] augmented the BART architecture with graph attention net-
works to inject knowledge for commonsense reasoning. JointLK [233]
proposed a joint reasoning framework connecting language models
and KGs through a shared semantic space. Wang et al. [248] unified
structure reasoning with pre-training by creating cross-modal repre-
sentations. GreasLM [293] enhanced KG integration by introducing
graph-aware semantic aggregation that improved question answer-
ing over heterogeneous knowledge sources. Dragon [275] took a dif-
ferent approach with deep bidirectional language-knowledge graph
pre-training that created entity-aware representations. (2) Rule-
based Integration. Complementing neural approaches, rule-based
methods used explicit reasoning paths over KGs. MHGRN [56]
implemented multi-hop graph relation networks that performed
explicit message passing between entities relevant to the query. QA-
GNN [276] addressed this by jointly reasoning over language and
KG representations, formulating paths through a graph neural net-
work architecture. UniKGQA [88] unified retrieval and reasoning
through transformers operating directly on knowledge subgraphs,
demonstrating better generalization to novel domains and question
types. These early approaches laid the foundation for more sophis-
ticated integration of structured knowledge with language models,
establishing key mechanisms for knowledge grounding and explicit
reasoning paths.
5.2.2
Modern Approaches (LLM + KG). Recent advancements show-
case stronger integration strategies between LLMs and KGs, moving
beyond early fusion methods to develop more sophisticated rea-
soning frameworks. These approaches enable LLMs to leverage
the structured representations in KGs more effectively, improving
reasoning capabilities while maintaining interpretability.
LLM reasoning over KG structure. ToG [230] enables LLMs to
reason directly with KGs through guided exploration. Starting from
key entities in the query, the model navigates the KG step-by-step
using prompt-driven reasoning to build explicit reasoning chains.
ToG-2 [158] tightly couples KG navigation with document contexts
through bidirectional iterative “knowledge-guided context retrieval”
and “context-enhanced graph retrieval”, enabling more precise rea-
soning. Besta et al. [12] introduce Graph-of-Thought (GoT), which
structures LLM reasoning as a graph rather than a linear chain. GoT
enables multi-path exploration, backtracking, and cyclical thinking
by representing reasoning steps as nodes and their relationships
as edges. Graph CoT [102] augments LLMs with graphs through
an iterative framework of reasoning, interaction, and execution.
The model identifies information needs, generates graph function
calls, and processes retrieved data sequentially, enabling step-by-
step traversal of complex graphs. RoG [152] introduces a planning-
retrieval-reasoning framework where LLMs first generate relation
paths grounded in KGs as plans, then retrieve valid reasoning paths
for structured reasoning. ChatRule [151] utilizes LLMs to mine
logical rules for KG reasoning, combining semantic understanding
with KG structure. It samples paths from KGs to prompt LLMs for
rule generation, ranks rules using confidence metrics, and applies
them for reasoning tasks without extensive rule enumeration, pre-
serving interpretability. MindMap [259] enables LLMs to construct
mind maps from KG inputs, combining path-based and neighbor-
based evidence subgraphs into reasoning graphs. The approach
helps LLMs perform synergistic inference by merging retrieved
KG facts with implicit knowledge, reducing hallucinations while
maintaining transparent reasoning pathways, particularly effective
in medical domains. ORT [146] introduces ontology-guided reverse
thinking for KGQA, constructing reasoning paths from query tar-
gets back to conditions using KG ontology. The approach extracts
both condition and target entities, builds backward reasoning trees,
and prunes irrelevant paths to guide precise knowledge retrieval.
KG-Embedded LLM is another track of research that aims to
directly incorporate KG information into the LLM’s architecture


KDD ’25, August 3–7, 2025, Toronto, ON, Canada
Pengcheng Jiang et al.
or training process, rather than using KGs as external resources
during inference time. This approach allows for more seamless
integration of structured knowledge within the generation pro-
cess. GraphToken [55] introduces a parameter-efficient method
to encode structured graph data for LLMs. Unlike text-based ap-
proaches, GraphToken learns an encoding function that generates
soft-token prompts to extend textual input with explicit structural
information. The system employs a graph neural network encoder
to capture graph structure while a projection layer aligns these
representations with the LLM’s embedding space. By freezing the
LLM parameters and only training the graph encoder, GraphToken
provides a computationally efficient way to enhance LLMs’ graph
reasoning capabilities. Building upon GraphToken, G-Retriever [76]
extends this approach by incorporating retrieval-augmented gener-
ation for improved textual graph understanding. While maintaining
the graph encoder architecture to align representations with the
LLM’s embedding space, G-Retriever addresses scaling challenges
by formulating subgraph retrieval as a Prize-Collecting Steiner Tree
optimization problem. This allows the system to select only the
most relevant portions of large graphs, enabling natural language
interaction with graphs through a conversational interface while
reducing hallucination in the generation.
Structure Summarization is another way to leverage knowledge
structures for LLM enhancement by condensing complex structured
knowledge into text formats. GraphRAG [50] organizes knowledge
through hierarchical community detection, creating multi-level
summaries of graph communities. Its map-reduce approach gener-
ates partial answers from relevant community summaries before ag-
gregating them into comprehensive responses, effectively handling
queries that require global corpus understanding. KARE [95] em-
ploys selective community retrieval using a multi-faceted relevance
scoring system that considers node overlap, coherence, recency,
and thematic relevance. Its dynamic retrieval algorithm prioritizes
diverse, valuable communities to augment contexts, enabling LLMs
to generate more accurate predictions and interpretable reasoning.
6
Future Directions
6.1
Technical Challenges
6.1.1
Retrieval Efficiency. Ensuring fast and accurate retrieval of
relevant information remains challenging as LLMs and data repos-
itories grow. Future work requires developing scalable indexing
methods for high-throughput requests while maintaining low la-
tency [85, 125]. Adaptive retrieval strategies—tailoring search depth
based on query complexity or user context—can enhance perfor-
mance and user satisfaction [2]. Addressing these needs holistically
is essential for real-world applications demanding both speed and
accuracy.
6.1.2
Knowledge Quality. The quality of structured representations
is critical, yet automation of taxonomy construction and informa-
tion organization can introduce noise or inconsistencies. Future
research should focus on robust validation techniques and iterative
refinement incorporating domain expertise [66]. Leveraging con-
textual signals—discourse patterns, domain ontologies, user feed-
back—can guide more fine-grained structuring, ensuring extracted
knowledge remains coherent across domains.
6.1.3
Integration Challenges. Integrating disparate knowledge sources
poses a key challenge for effective RAS systems. This process in-
volves reconciling conflicts, minimizing redundancy, and updating
representations as new information emerges. Incremental learning
and conflict-resolution strategies are crucial for maintaining consis-
tency as knowledge evolves [195]. Building adaptive frameworks
that can ingest and reconcile varied data sources in real time will
better equip LLMs to provide accurate, context-rich answers.
6.2
Research Opportunities
6.2.1
Multi-modal Knowledge Integration. As data becomes increas-
ingly diverse, future RAS solutions must handle images, video, and
audio alongside text. Unified multi-modal indexing strategies cap-
turing cross-modal semantic relationships can significantly enhance
information discovery [138]. Neural architectures combining vision-
language models with text encoders can facilitate cross-modal rea-
soning, allowing LLMs to reference non-textual cues in responses.
6.2.2
Cross-lingual Systems. Techniques leveraging multilingual
embeddings and parallel corpora can build language-agnostic tax-
onomies enabling global knowledge sharing. Transfer learning
methods reusing structured representations from high-resource
languages can accelerate development in low-resource languages
[2]. Standardizing cross-lingual benchmarks with robust domain
adaptation will drive accuracy across multilingual contexts[33].
6.2.3
Interactive and Self-Refining Systems. Conversational inter-
faces allow users to refine queries naturally, leading to more precise
retrieval. Self-refinement mechanisms using reinforcement learn-
ing or meta-learning enable systems to identify and correct errors
autonomously, mitigating hallucinations and outdated knowledge
[48]. This combined approach improves both accuracy and explain-
ability as models document their corrective steps while adapting to
user feedback[6, 221].
6.2.4
Human-AI Collaborative Frameworks. While automation is
central to RAS, human expertise remains essential for complex tasks.
Expert reviewers can guide taxonomy refinement, while crowd-
sourcing methods help maintain data relevance over time [176].
These collaborative frameworks ensure models adhere to privacy
regulations and fairness principles, fostering trustworthy solutions
that leverage both machine scalability and human judgment[190].
6.2.5
Personalized Knowledge Delivery. Tailoring retrieval and struc-
turing processes to individual users significantly enhances rele-
vance. Personalized approaches incorporate user profiles, historical
behavior, or explicit feedback to shape retrieval strategies [142],
while privacy-preserving techniques safeguard personal data. By
layering contextual cues on core architectures, these systems can de-
liver insights that evolve with user needs, providing more engaging
interactions [65, 94].
7
Conclusions
This survey presented Retrieval-And-Structuring (RAS) as a para-
digm that enhances LLMs by integrating dynamic retrieval with
structured knowledge to mitigate hallucinations, outdated informa-
tion, and limited domain expertise. We reviewed key techniques
in retrieval, text structuring, and their integration with LLMs for


A Survey on Retrieval And Structuring Augmented Generation with Large Language Models
KDD ’25, August 3–7, 2025, Toronto, ON, Canada
knowledge-grounded generation. Despite progress, challenges re-
main in retrieval efficiency, knowledge quality, and integration.
Promising directions include multimodal and cross-lingual systems,
self-refining frameworks, human-AI collaboration, and personal-
ized knowledge delivery—advancing LLM reasoning with factual
and current knowledge across domains.
Acknowledgment
Research was supported in part by National Science Foundation
IIS-19-56151, the Molecule Maker Lab Institute: An AI Research
Institutes program supported by NSF under Award No. 2019897,
and the Institute for Geospatial Understanding through an Integra-
tive Discovery Environment (I-GUIDE) by NSF under Award No.
2118329, US DARPA INCAS Program No. HR0011-21-C0165 and
BRIES Program No. HR0011-24-3-0325.
References
[1] Zahra Abbasiantaeb and Saeedeh Momtazi. 2021. Text-based question answering
from information retrieval and deep neural network perspectives: A survey.
Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 11, 6
(2021), e1412.
[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774
(2023).
[3] Anirudh Ajith, Mengzhou Xia, Alexis Chevalier, Tanya Goyal, Danqi Chen, and
Tianyu Gao. 2024. Litsearch: A retrieval benchmark for scientific literature
search. arXiv preprint arXiv:2407.18940 (2024).
[4] Negar Arabzadeh, Xinyi Yan, and Charles LA Clarke. 2021. Predicting effi-
ciency/effectiveness trade-offs for dense vs. sparse retrieval strategy selection.
In Proceedings of the 30th ACM International Conference on Information & Knowl-
edge Management. 2862–2866.
[5] Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Ho-
jel, Immanuel Trummer, and Christopher Ré. 2023. Language Models Enable
Simple Systems for Generating Structured Views of Heterogeneous Data Lakes.
Proceedings of the VLDB Endowment 17, 2 (2023), 92–105.
[6] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-rag: Learning to retrieve, generate, and critique through self-reflection. In
The Twelfth International Conference on Learning Representations.
[7] Hiteshwar Kumar Azad and Akshay Deepak. 2019. Query expansion techniques
for information retrieval: A survey. Information Processing & Management 56, 5
(2019), 1698–1735. doi:10.1016/j.ipm.2019.05.009
[8] Sonia Badene, Kate Thompson, Jean-Pierre Lorré, and Nicholas Asher. 2019.
Data Programming for Learning Discourse Structure. In ACL.
[9] Siddhartha Banerjee, Cem Akkaya, Francisco Perez-Sorrosal, and Kostas Tsiout-
siouliklis. 2019. Hierarchical Transfer Learning for Multi-label Text Classifica-
tion. In ACL.
[10] Petr Baudiš and Jan Šediv`y. 2015. Modeling of the question answering task
in the yodaqa system. In Experimental IR Meets Multilinguality, Multimodality,
and Interaction: 6th International Conference of the CLEF Association, CLEF’15,
Toulouse, France, September 8-11, 2015, Proceedings 6. Springer, 222–228.
[11] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic
parsing on freebase from question-answer pairs. In Proceedings of the 2013
conference on empirical methods in natural language processing. 1533–1544.
[12] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski,
Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr
Nyczyk, and Torsten Hoefler. 2024. Graph of Thoughts: Solving Elaborate
Problems with Large Language Models. Proceedings of the AAAI Conference on
Artificial Intelligence 38, 16 (Mar. 2024), 17682–17690. doi:10.1609/aaai.v38i16.
29720
[13] Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Scott Yih, Sebastian
Riedel, and Fabio Petroni. 2022. Autoregressive search engines: Generating
substrings as document identifiers. Advances in NeurIPS 35 (2022), 31668–31683.
[14] Vladimir Blagojevi. 2023. Enhancing rag pipelines in haystack: Introducing
diversityranker and lostinthemiddleranker.
[15] Luiz Bonifacio, Hugo Abonizio, Marzieh Fadaee, and Rodrigo Nogueira. 2022.
Inpars: Data augmentation for information retrieval using large language models.
arXiv preprint arXiv:2202.05144 (2022).
[16] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bog-
dan Damoc, Aidan Clark, et al. 2022. Improving language models by retrieving
from trillions of tokens. In International conference on machine learning. PMLR,
2206–2240.
[17] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot learners. Advances in NeurIPS
33 (2020), 1877–1901.
[18] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter,
Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners.
In NeurIPS.
[19] Claudio Carpineto and Giovanni Romano. 2012. A Survey of Automatic Query
Expansion in Information Retrieval. ACM Comput. Surv. 44, 1, Article 1 (Jan.
2012), 50 pages. doi:10.1145/2071389.2071390
[20] Oishik Chatterjee, Ganesh Ramakrishnan, and Sunita Sarawagi. 2018. Data
Programming using Continuous and Quality-Guided Labeling Functions. In
AAAI.
[21] Aditi Chaudhary, Karthik Raman, and Michael Bendersky. 2023. It’s All Relative!–
A Synthetic Query Generation Approach for Improving Zero-Shot Relevance
Prediction. arXiv preprint arXiv:2311.07930 (2023).
[22] Aditi Chaudhary, Jiateng Xie, Zaid Sheikh, Graham Neubig, and Jaime Carbonell.
2019. A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-
resource Named Entity Recognizers. In EMNLP-IJCNLP. 5164–5174.
[23] Haibin Chen, Qianli Ma, Zhenxi Lin, and Jiangyue Yan. 2021. Hierarchy-aware
Label Semantics Matching Network for Hierarchical Text Classification. In
ACL-IJCNLP.
[24] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large
language models in retrieval-augmented generation. In Proceedings of the AAAI
Conference on Artificial Intelligence, Vol. 38. 17754–17762.
[25] Jiawei Chen, Qing Liu, Hongyu Lin, Xianpei Han, and Le Sun. 2022. Few-shot
Named Entity Recognition with Self-describing Networks. In ACL. 5711–5722.
[26] Jiao Chen, Luyi Ma, Xiaohan Li, Nikhil Thakurdesai, Jianpeng Xu, Jason HD
Cho, Kaushiki Nag, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2023.
Knowledge graph completion models are few-shot learners: An empirical study
of relation labeling in e-commerce with llms. arXiv preprint arXiv:2305.09858
(2023).
[27] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020.
A simple framework for contrastive learning of visual representations. In Inter-
national conference on machine learning. PmLR, 1597–1607.
[28] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao,
Hongming Zhang, and Dong Yu. 2024. Dense x retrieval: What retrieval granu-
larity should we use?. In Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing. 15159–15177.
[29] Xiang Chen, Ningyu Zhang, Xin Xie, Shumin Deng, Yunzhi Yao, Chuanqi Tan,
Fei Huang, Luo Si, and Huajun Chen. 2022. KnowPrompt: Knowledge-aware
Prompt-tuning with Synergistic Optimization for Relation Extraction. In WWW.
2778–2788.
[30] Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan.
2023. Lift yourself up: Retrieval-augmented text generation with self-memory.
Advances in NeurIPS 36 (2023), 43780–43799.
[31] Yiruo Cheng, Kelong Mao, and Zhicheng Dou. 2024. Interpreting Conversational
Dense Retrieval by Rewriting-Enhanced Inversion of Session Embedding. arXiv
preprint arXiv:2402.12774 (2024).
[32] Yew Ken Chia, Lidong Bing, Soujanya Poria, and Luo Si. 2022. RelationPrompt:
Leveraging Prompts to Generate Synthetic Data for Zero-Shot Relation Triplet
Extraction. In Findings of the Association for Computational Linguistics: ACL
2022, Dublin, Ireland, May 22-27, 2022, Smaranda Muresan, Preslav Nakov, and
Aline Villavicencio (Eds.). Association for Computational Linguistics, 45–57.
doi:10.18653/V1/2022.FINDINGS-ACL.5
[33] Nadezhda Chirkova, David Rau, Hervé Déjean, Thibault Formal, Stéphane Clin-
chant, and Vassilina Nikoulina. 2024. Retrieval-augmented generation in multi-
lingual settings. arXiv preprint arXiv:2407.01463 (2024).
[34] Eunsol Choi, Omer Levy, Yejin Choi, and Luke Zettlemoyer. 2018. Ultra-Fine
Entity Typing. In ACL. 87–96.
[35] Yung-Sung Chuang, Wei Fang, Shang-Wen Li, Wen-tau Yih, and James Glass.
2023. Expand, rerank, and retrieve: Query reranking for open-domain question
answering. arXiv preprint arXiv:2305.17080 (2023).
[36] Robert Churchill, Lisa Singh, Rebecca Ryan, and Pamela Davis-Kean. 2022. A
Guided Topic-Noise Model for Short Texts. In WWW’22.
[37] John Dagdelen, Alexander Dunn, Sanghoon Lee, Nicholas Walker, Andrew S
Rosen, Gerbrand Ceder, Kristin A Persson, and Anubhav Jain. 2024. Structured
information extraction from scientific text with large language models. Nature
Communications 15, 1 (2024), 1418.
[38] Hongliang Dai, Yangqiu Song, and Haixun Wang. 2021. Ultra-Fine Entity Typing
with Weak Supervision from a Masked Language Model. In ACL’21. 1790–1799.


KDD ’25, August 3–7, 2025, Toronto, ON, Canada
Pengcheng Jiang et al.
[39] Zhuyun Dai and Jamie Callan. 2019. Context-aware sentence/passage term
importance estimation for first stage retrieval. arXiv preprint arXiv:1910.10687
(2019).
[40] Zhuyun Dai, Vincent Y Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov,
Kelvin Guu, Keith B Hall, and Ming-Wei Chang. 2022. Promptagator: Few-shot
dense retrieval from 8 examples. arXiv preprint arXiv:2209.11755 (2022).
[41] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2020. Au-
toregressive entity retrieval. arXiv preprint arXiv:2010.00904 (2020).
[42] DeepSeek-AI. 2024. DeepSeek-V3 Technical Report. CoRR abs/2412.19437 (2024).
arXiv:2412.19437 doi:10.48550/ARXIV.2412.19437
[43] DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning. CoRR abs/2501.12948 (2025). arXiv:2501.12948
doi:10.48550/ARXIV.2501.12948
[44] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert:
Pre-training of deep bidirectional transformers for language understanding. In
Proceedings of the 2019 conference of the North American chapter of the association
for computational linguistics: human language technologies, volume 1 (long and
short papers). 4171–4186.
[45] Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. 2020. Topic Modeling in
Embedding Spaces. TACL (2020).
[46] Linyi Ding, Sizhe Zhou, Jinfeng Xiao, and Jiawei Han. 2024. Automated con-
struction of theme-specific knowledge graphs. arXiv preprint arXiv:2404.19146
(2024).
[47] Linyi Ding, Sizhe Zhou, Jinfeng Xiao, and Jiawei Han. 2024. Automated Con-
struction of Theme-specific Knowledge Graphs. CoRR abs/2404.19146 (2024).
arXiv:2404.19146 doi:10.48550/ARXIV.2404.19146
[48] Jiannan Dong, Tao Wang, Yicheng Zheng, Yijun Wu, Jiefu Wei, Ming Ding,
Xipeng Qiu, and Jie Tang. 2023. Self-Refine: Iterative Refinement with GPT-4.
arXiv preprint arXiv:2303.17651 (2023). https://arxiv.org/abs/2303.17651.
[49] Christopher Dozier, Ravikumar Kondadadi, Marc Light, Arun Vachher, Sriharsha
Veeramachaneni, and Ramdev Wudali. 2010. Named entity recognition and
resolution in legal text. Springer.
[50] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130
(2024).
[51] Fartash Faghri, David J Fleet, Jamie Ryan Kiros, and Sanja Fidler. 2017. Vse++:
Improving visual-semantic embeddings with hard negatives. arXiv preprint
arXiv:1707.05612 (2017).
[52] Christos Faloutsos and Douglas W Oard. 1995. A survey of information retrieval
and filtering methods. Citeseer.
[53] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Proceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining. 6491–6501.
[54] Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and
Yiqun Liu. 2024. Scaling laws for dense retrieval. In Proceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 1339–1349.
[55] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. 2024. Talk like a Graph:
Encoding Graphs for Large Language Models. In The Twelfth International
Conference on Learning Representations.
https://openreview.net/forum?id=
IuXR1CCrSi
[56] Yanlin Feng, Xinyue Chen, Bill Yuchen Lin, Peifeng Wang, Jun Yan, and Xi-
ang Ren. 2020. Scalable multi-hop relational reasoning for knowledge-aware
question answering. arXiv preprint arXiv:2005.00646 (2020).
[57] Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric Lenssen, Rishabh Ranjan,
Joshua Robinson, Rex Ying, Jiaxuan You, and Jure Leskovec. 2023. Relational
deep learning: Graph representation learning on relational databases. arXiv
preprint arXiv:2312.04615 (2023).
[58] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clin-
chant. 2021. SPLADE v2: Sparse lexical and expansion model for information
retrieval. arXiv preprint arXiv:2109.10086 (2021).
[59] Chunjing Gan, Dan Yang, Binbin Hu, Ziqi Liu, Yue Shen, Zhiqiang Zhang, Jinjie
Gu, Jun Zhou, and Guannan Zhang. 2023. Making Large Language Models
Better Knowledge Miners for Online Marketing with Progressive Prompting
Augmentation. arXiv preprint arXiv:2312.05276 (2023).
[60] Jingtong Gao, Bo Chen, Xiangyu Zhao, Weiwen Liu, Xiangyang Li, Yichao Wang,
Zijian Zhang, Wanyu Wang, Yuyang Ye, Shanru Lin, et al. 2024. Llm-enhanced
reranking in recommender systems. arXiv preprint arXiv:2406.12433 (2024).
[61] Luyu Gao and Jamie Callan. 2021. Condenser: a pre-training architecture for
dense retrieval. arXiv preprint arXiv:2104.08253 (2021).
[62] Luyu Gao and Jamie Callan. 2021. Unsupervised corpus aware language model
pre-training for dense passage retrieval. arXiv preprint arXiv:2108.05540 (2021).
[63] Luyu Gao, Zhuyun Dai, Tongfei Chen, Zhen Fan, Benjamin Van Durme, and
Jamie Callan. 2021. Complement lexical retrieval model with semantic residual
embeddings. In Advances in Information Retrieval: 43rd European Conference on
IR Research, ECIR 2021, Virtual Event, March 28–April 1, 2021, Proceedings, Part I
43. Springer, 146–160.
[64] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey. arXiv preprint arXiv:2312.10997
2 (2023).
[65] Shijie Geng, Zuohui Fu, Yingqiang Ge, Lei Li, Gerard De Melo, and Yongfeng
Zhang. 2022. Improving personalized explanation generation through visualiza-
tion. In Proceedings of the 60th Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers). 244–255.
[66] Google Research. 2023. Introducing PaLM 2. Blog post, https://ai.googleblog.
com/2023/05/google-introduces-palm-2.html. Accessed: 2023-XX-XX.
[67] Jiafeng Guo, Yinqiong Cai, Yixing Fan, Fei Sun, Ruqing Zhang, and Xueqi Cheng.
2022. Semantic models for the first-stage retrieval: A comprehensive review.
ACM Transactions on Information Systems (TOIS) 40, 4 (2022), 1–42.
[68] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu
Su. 2024. Hipporag: Neurobiologically inspired long-term memory for large
language models. In The Thirty-eighth Annual Conference on NeurIPS.
[69] Rishin Haldar and Debajyoti Mukhopadhyay. 2011. Levenshtein distance tech-
nique in dictionary lookup methods: An improved approach. arXiv preprint
arXiv:1101.1232 (2011).
[70] Kailash A Hambarde and Hugo Proenca. 2023. Information retrieval: recent
advances and beyond. IEEE Access 11 (2023), 76581–76604.
[71] Xu Han, Weilin Zhao, Ning Ding, Zhiyuan Liu, and Maosong Sun. 2022. Ptr:
Prompt tuning with rules for text classification. AI Open 3 (2022), 182–192.
[72] Bahareh Harandizadeh, J. Hunter Priniski, and Fred Morstatter. 2022. Keyword
Assisted Embedded Topic Model. In WSDM’22.
[73] Taher H. Haveliwala. 2002. Topic-sensitive PageRank. In Proceedings of the 11th
International Conference on World Wide Web (Honolulu, Hawaii, USA) (WWW
’02). Association for Computing Machinery, New York, NY, USA, 517–526. doi:10.
1145/511446.511513
[74] Shirley Anugrah Hayati, Raphael Olivier, Pravalika Avvaru, Pengcheng Yin,
Anthony Tomasic, and Graham Neubig. 2018. Retrieval-based neural code
generation. arXiv preprint arXiv:1808.10025 (2018).
[75] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2020. Momen-
tum contrast for unsupervised visual representation learning. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition. 9729–9738.
[76] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann Le-
Cun, Xavier Bresson, and Bryan Hooi. 2025. G-retriever: Retrieval-augmented
generation for textual graph understanding and question answering. Advances
in NeurIPS 37 (2025), 132876–132907.
[77] Wilbert Jan Heeringa. 2004. Measuring dialect pronunciation differences using
Levenshtein distance. (2004).
[78] Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo Zhao, and Hang Zhao.
2023. ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory.
CoRR abs/2306.03901 (2023). arXiv:2306.03901 doi:10.48550/ARXIV.2306.03901
[79] Jiaxin Huang, Yu Meng, and Jiawei Han. 2022. Few-Shot Fine-Grained Entity
Typing with Automatic Label Interpretation and Instance Generation. In KDD.
605–614.
[80] Jiaxin Huang, Yiqing Xie, Yu Meng, Jiaming Shen, Yunyi Zhang, and Jiawei Han.
2020. Guiding Corpus-based Set Expansion by Auxiliary Sets Generation and
Co-Expansion. In WWW. 2188–2198.
[81] Jiaxin Huang, Yiqing Xie, Yu Meng, Yunyi Zhang, and Jiawei Han. 2020. CoRel:
Seed-Guided Topical Taxonomy Construction by Concept Learning and Relation
Transferring. In KDD’20.
[82] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2025.
A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions. ACM Transactions on Information Systems 43, 2
(2025), 1–55.
[83] Pere-Lluís Huguet Cabot and Roberto Navigli. 2021. REBEL: Relation Extraction
By End-to-end Language generation. In Findings of EMNLP. 2370–2381.
[84] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning. arXiv preprint arXiv:2112.09118
(2021).
[85] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models.
Journal of Machine Learning Research 24, 251 (2023), 1–43.
[86] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination
in natural language generation. ACM computing surveys 55, 12 (2023), 1–38.
[87] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin Zhao, and Ji-Rong Wen.
2023. StructGPT: A General Framework for Large Language Model to Reason
over Structured Data. In Proceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023,
Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computa-
tional Linguistics, 9237–9251. doi:10.18653/V1/2023.EMNLP-MAIN.574


A Survey on Retrieval And Structuring Augmented Generation with Large Language Models
KDD ’25, August 3–7, 2025, Toronto, ON, Canada
[88] Jinhao Jiang, Kun Zhou, Xin Zhao, and Ji-Rong Wen. 2023. UniKGQA: Uni-
fied Retrieval and Reasoning for Solving Multi-hop Question Answering Over
Knowledge Graph. In The Eleventh International Conference on Learning Repre-
sentations.
[89] Pengcheng Jiang, Shivam Agarwal, Bowen Jin, Xuan Wang, Jimeng Sun, and
Jiawei Han. 2023. Text Augmented Open Knowledge Graph Completion via
Pre-Trained Language Models. In Findings of ACL. 11161–11180.
[90] Pengcheng Jiang, Lang Cao, Cao Danica Xiao, Parminder Bhatia, Jimeng Sun,
and Jiawei Han. 2025. Kg-fit: Knowledge graph fine-tuning upon open-world
knowledge. Advances in NeurIPS 37 (2025), 136220–136258.
[91] Pengcheng Jiang, Megan Amber Lim, Adam Cross, and Jimeng Sun. [n. d.].
MedKG: Empowering Medical Education with Interactive Construction and
Visualization of Knowledge Graphs via Large Language Models. ([n. d.]).
[92] Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng
Wang, Jimeng Sun, and Jiawei Han. 2025. Deepretrieval: Hacking real search
engines and retrievers with large language models via reinforcement learning.
arXiv preprint arXiv:2503.00223 (2025).
[93] Pengcheng Jiang, Jiacheng Lin, Zifeng Wang, Jimeng Sun, and Jiawei Han.
2024. GenRES: Rethinking Evaluation for Generative Relation Extraction in
the Era of Large Language Models. In Proceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and
Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City,
Mexico, 2820–2837. doi:10.18653/v1/2024.naacl-long.155
[94] Pengcheng Jiang, Cao Xiao, Adam Cross, and Jimeng Sun. 2023. Graphcare:
Enhancing healthcare predictions with personalized knowledge graphs. arXiv
preprint arXiv:2305.12788 (2023).
[95] Pengcheng Jiang, Cao Xiao, Minhao Jiang, Parminder Bhatia, Taha Kass-Hout,
Jimeng Sun, and Jiawei Han. 2025. Reasoning-Enhanced Healthcare Predictions
with Knowledge Graph Community Retrieval. In The Thirteenth International
Conference on Learning Representations.
https://openreview.net/forum?id=
8fLgt7PQza
[96] Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng
Sun, and Jiawei Han. 2025. s3: You Don’t Need That Much Data to Train a Search
Agent via RL. arXiv:2505.14146 [cs.AI]
[97] Ting Jiang, Deqing Wang, Leilei Sun, Zhongzhi Chen, Fuzhen Zhuang, and
Qinghong Yang. 2022. Exploiting Global and Local Hierarchies for Hierarchical
Text Classification. In EMNLP. 4030–4039.
[98] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval aug-
mented generation. In Proceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing. 7969–7992.
[99] Yizhu Jiao, Sha Li, Yiqing Xie, Ming Zhong, Heng Ji, and Jiawei Han. 2022.
Open-Vocabulary Argument Role Prediction For Event Extraction. In Findings
of EMNLP. 5404–5418.
[100] Yizhu Jiao, Ming Zhong, Sha Li, Ruining Zhao, Siru Ouyang, Heng Ji, and Jiawei
Han. 2023. Instruct and Extract: Instruction Tuning for On-Demand Information
Extraction. In EMNLP. 10030–10051.
[101] Yizhu Jiao, Ming Zhong, Jiaming Shen, Yunyi Zhang, Chao Zhang, and Jiawei
Han. 2023. Unsupervised Event Chain Mining from Multiple Documents. In
Proceedings of the ACM Web Conference 2023, WWW 2023, Austin, TX, USA, 30
April 2023 - 4 May 2023, Ying Ding, Jie Tang, Juan F. Sequeda, Lora Aroyo, Carlos
Castillo, and Geert-Jan Houben (Eds.). ACM, 1948–1959. doi:10.1145/3543507.
3583295
[102] Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Kumar Roy, Yu Zhang, Zheng Li,
Ruirui Li, Xianfeng Tang, Suhang Wang, Yu Meng, et al. 2024. Graph Chain-
of-Thought: Augmenting Large Language Models by Reasoning on Graphs. In
Findings of the Association for Computational Linguistics ACL 2024. 163–184.
[103] Zhi Jing, Yongye Su, Yikun Han, Bo Yuan, Haiyun Xu, Chunjiang Liu, Kehai Chen,
and Min Zhang. 2024. When Large Language Models Meet Vector Databases:
A Survey. CoRR abs/2402.01763 (2024). arXiv:2402.01763 doi:10.48550/ARXIV.
2402.01763
[104] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa:
A large scale distantly supervised challenge dataset for reading comprehension.
arXiv preprint arXiv:1705.03551 (2017).
[105] Hao Kang, Tevin Wang, and Chenyan Xiong. 2024. Interpret and Control Dense
Retrieval with Sparse Latent Features. arXiv preprint arXiv:2411.00786 (2024).
[106] Minki Kang, Jin Myung Kwak, Jinheon Baek, and Sung Ju Hwang. 2023. Knowl-
edge graph-augmented language models for knowledge-grounded dialogue
generation. arXiv preprint arXiv:2305.18846 (2023).
[107] SeongKu Kang, Shivam Agarwal, Bowen Jin, Dongha Lee, Hwanjo Yu, and Jiawei
Han. 2024. Improving retrieval in theme-specific applications using a corpus
topical taxonomy. In Proceedings of the ACM Web Conference 2024. 1497–1508.
[108] SeongKu Kang, Yunyi Zhang, Pengcheng Jiang, Dongha Lee, Jiawei Han, and
Hwanjo Yu. 2024. Taxonomy-guided Semantic Indexing for Academic Paper
Search. In Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen
(Eds.). Association for Computational Linguistics, Miami, Florida, USA, 7169–
7184. doi:10.18653/v1/2024.emnlp-main.407
[109] Priyanka Kargupta, Yunyi Zhang, Yizhu Jiao, Siru Ouyang, and Jiawei Han. 2024.
Unsupervised Episode Detection for Large-Scale News Events. arXiv preprint
arXiv:2408.04873 (2024).
[110] Md Rezaul Karim, Lina Molinas Comet, Md Shajalal, Oya Beyan, Dietrich
Rebholz-Schuhmann, and Stefan Decker. 2023. From Large Language Mod-
els to Knowledge Graphs for Biomarker Discovery in Cancer. arXiv preprint
arXiv:2310.08365 (2023).
[111] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu,
Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval
for Open-Domain Question Answering.. In EMNLP (1). 6769–6781.
[112] Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang,
Christopher Potts, and Matei Zaharia. 2022. Demonstrate-search-predict: Com-
posing retrieval and language models for knowledge-intensive nlp.
arXiv
preprint arXiv:2212.14024 (2022).
[113] Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage
search via contextualized late interaction over bert. In Proceedings of the 43rd
International ACM SIGIR conference on research and development in Information
Retrieval. 39–48.
[114] Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joonsuk Park, and Jaewoo Kang.
2023. Tree of clarifications: Answering ambiguous questions with retrieval-
augmented large language models. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing. 996–1009.
[115] Mei Kobayashi and Koichi Takeda. 2000. Information retrieval on the web. ACM
computing surveys (CSUR) 32, 2 (2000), 144–173.
[116] Tanay Komarlu, Minhao Jiang, Xuan Wang, and Jiawei Han. 2023. OntoType:
Ontology-Guided Zero-Shot Fine-Grained Entity Typing with Weak Supervision
from Pre-Trained Language Models. arXiv preprint arXiv:2305.12307 (2023).
[117] Rajasekar Krishnamurthy, Sriram Raghavan, Shivakumar Vaithyanathan, and
Huaiyu Zhu. 2006. Avatar information extraction system. (2006).
[118] Keito Kudo, Hiroyuki Deguchi, Makoto Morishita, Ryo Fujii, Takumi Ito, Shin-
taro Ozaki, Koki Natsumi, Kai Sato, Kazuki Yano, Ryosuke Takahashi, et al. 2024.
Document-level translation with LLM reranking: Team-j at WMT 2024 general
translation task. In Proceedings of the Ninth Conference on Machine Translation.
210–226.
[119] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, An-
drew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions:
A Benchmark for Question Answering Research. Transactions of the Association
for Computational Linguistics 7 (2019), 452–466. doi:10.1162/tacl_a_00276
[120] Dongha Lee, Jiaming Shen, SeongKu Kang, Susik Yoon, Jiawei Han, and Hwanjo
Yu. 2022. TaxoCom: Topic Taxonomy Completion with Hierarchical Discovery
of Novel Topic Clusters. In WWW’22.
[121] Elena Leitner, Georg Rehm, and Julian Moreno-Schneider. 2019. Fine-grained
named entity recognition in legal documents. In International conference on
semantic systems. Springer, 272–287.
[122] Dirk Lewandowski. 2005. Web searching, search engines and Information
Retrieval. Information Services & Use 25, 3-4 (2005), 137–147.
[123] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman
Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART:
Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
Translation, and Comprehension. In ACL. 7871–7880.
[124] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive
nlp tasks. Advances in NeurIPS 33 (2020), 9459–9474.
[125] Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Minervini, Heinrich Küttler,
Aleksandra Piktus, Pontus Stenetorp, and Sebastian Riedel. 2021. Paq: 65 million
probably-asked questions and what you can do with them. Transactions of the
Association for Computational Linguistics 9 (2021), 1098–1115.
[126] Bangzheng Li, Wenpeng Yin, and Muhao Chen. 2022. Ultra-fine entity typing
with indirect supervision from natural language inference. TACL 10 (2022),
607–622.
[127] Guozheng Li, Peng Wang, and Wenjun Ke. 2023. Revisiting large language
models as zero-shot relation extractors. arXiv preprint arXiv:2310.05028 (2023).
[128] Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang,
Bowen Qin, Ruiying Geng, Nan Huo, et al. 2023. Can llm already serve as a
database interface? a big bench for large-scale database grounded text-to-sqls.
Advances in NeurIPS 36 (2023), 42330–42357.
[129] Junpeng Li, Zixia Jia, and Zilong Zheng. 2023. Semi-automatic Data Enhance-
ment for Document-Level Relation Extraction with Distant Supervision from
Large Language Models. In Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10,
2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Com-
putational Linguistics, 5495–5505. doi:10.18653/V1/2023.EMNLP-MAIN.334
[130] Jing Li, Aixin Sun, Jianglei Han, and Chenliang Li. 2020. A survey on deep
learning for named entity recognition. IEEE transactions on knowledge and data


KDD ’25, August 3–7, 2025, Toronto, ON, Canada
Pengcheng Jiang et al.
engineering 34, 1 (2020), 50–70.
[131] Minghan Li and Eric Gaussier. 2024. Domain Adaptation for Dense Retrieval
and Conversational Dense Retrieval through Self-Supervision by Meticulous
Pseudo-Relevance Labeling. arXiv preprint arXiv:2403.08970 (2024).
[132] Na Li, Zied Bouraoui, and Steven Schockaert. 2023. Ultra-fine entity typing
with prior knowledge about labels: A simple clustering based strategy. arXiv
preprint arXiv:2305.12802 (2023).
[133] Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang,
Danielle Rifinski Fainman, Dongmei Zhang, and Surajit Chaudhuri. 2023. Table-
gpt: Table-tuned gpt for diverse table tasks. arXiv preprint arXiv:2310.09263
(2023).
[134] Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, and
Zhicheng Dou. 2024. From matching to generation: A survey on generative
information retrieval. arXiv preprint arXiv:2404.14851 (2024).
[135] Xianming Li and Jing Li. 2023. Angle-optimized text embeddings. arXiv preprint
arXiv:2309.12871 (2023).
[136] Xinze Li, Zhenghao Liu, Chenyan Xiong, Shi Yu, Yu Gu, Zhiyuan Liu, and Ge
Yu. 2023. Structure-aware language model pretraining improves dense retrieval
on structured data. arXiv preprint arXiv:2305.19912 (2023).
[137] Yinghui Li, Yangning Li, Yuxin He, Tianyu Yu, Ying Shen, and Hai-Tao Zheng.
2022. Contrastive Learning with Hard Negative Entities for Entity Set Expansion.
In SIGIR. 1077–1086.
[138] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and
Ke Liu. 2024. A Survey of Multimodel Large Language Models. In Proceedings of
the 3rd International Conference on Computer, Artificial Intelligence and Control
Engineering. 405–409.
[139] Jimmy Lin and Xueguang Ma. 2021. A few brief notes on deepimpact, coil, and
a conceptual framework for information retrieval techniques. arXiv preprint
arXiv:2106.14807 (2021).
[140] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard
James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, et al. 2023.
Ra-dit: Retrieval-augmented dual instruction tuning. In The Twelfth International
Conference on Learning Representations.
[141] Xi Victoria Lin, Richard Socher, and Caiming Xiong. 2020. Bridging textual
and tabular data for cross-domain text-to-SQL semantic parsing. arXiv preprint
arXiv:2012.12627 (2020).
[142] Fang Liu, Clement Yu, and Weiyi Meng. 2004. Personalized web search for
improving retrieval effectiveness. IEEE Transactions on knowledge and data
engineering 16, 1 (2004), 28–40.
[143] Jie Liu and Barzan Mozafari. 2024. Query rewriting via large language models.
arXiv preprint arXiv:2403.09060 (2024).
[144] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models
use long contexts. Transactions of the Association for Computational Linguistics
12 (2024), 157–173.
[145] Nelson F Liu, Tianyi Zhang, and Percy Liang. 2023. Evaluating verifiability in
generative search engines. arXiv preprint arXiv:2304.09848 (2023).
[146] Runxuan Liu, Bei Luo, Jiaqi Li, Baoxin Wang, Ming Liu, Dayong Wu, Shijin
Wang, and Bing Qin. 2025. Ontology-Guided Reverse Thinking Makes Large
Language Models Stronger on Knowledge Graph Question Answering. arXiv
preprint arXiv:2502.11491 (2025).
[147] Ye Liu, Yao Wan, Lifang He, Hao Peng, and S Yu Philip. 2021. Kg-bart: Knowledge
graph-augmented bart for generative commonsense reasoning. In Proceedings
of the AAAI conference on artificial intelligence, Vol. 35. 6418–6425.
[148] Michael Llordes, Debasis Ganguly, Sumit Bhatia, and Chirag Agarwal. 2023.
Explain like I am BM25: Interpreting a Dense Model’s Ranked-List with a Sparse
Approximation. In Proceedings of the 46th International ACM SIGIR Conference
on Research and Development in Information Retrieval. 1976–1980.
[149] Keming Lu, I Hsu, Wenxuan Zhou, Mingyu Derek Ma, Muhao Chen, et al. 2022.
Summarization as indirect supervision for relation extraction. arXiv preprint
arXiv:2205.09837 (2022).
[150] Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse,
dense, and attentional representations for text retrieval. Transactions of the
Association for Computational Linguistics 9 (2021), 329–345.
[151] Linhao Luo, Jiaxin Ju, Bo Xiong, Yuan-Fang Li, Gholamreza Haffari, and Shirui
Pan. 2023. Chatrule: Mining logical rules with large language models for knowl-
edge graph reasoning. arXiv preprint arXiv:2309.01538 (2023).
[152] LINHAO LUO, Yuan-Fang Li, Reza Haf, and Shirui Pan. 2024. Reasoning on
Graphs: Faithful and Interpretable Large Language Model Reasoning. In The
Twelfth International Conference on Learning Representations. https://openreview.
net/forum?id=ZGNWW7xZ6Q
[153] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and
Shirui Pan. 2025. GFM-RAG: Graph Foundation Model for Retrieval Augmented
Generation. arXiv preprint arXiv:2502.01113 (2025).
[154] Ziyang Luo, Can Xu, Pu Zhao, Xiubo Geng, Chongyang Tao, Jing Ma, Qingwei
Lin, and Daxin Jiang. 2023. Augmented large language models with parametric
knowledge guiding. arXiv preprint arXiv:2305.04757 (2023).
[155] Yuanhua Lv and ChengXiang Zhai. 2011. Adaptive Term Frequency Normal-
ization for BM25. In Proceedings of the 20th ACM International Conference on
Information and Knowledge Management (CIKM 2011). 1985–1988.
[156] Yuanhua Lv and ChengXiang Zhai. 2011. Lower-bounding term frequency
normalization. In Proceedings of the 20th ACM international conference on Infor-
mation and knowledge management. 7–16.
[157] Mingyu Derek Ma, Xiaoxuan Wang, Po-Nien Kung, P. Jeffrey Brantingham,
Nanyun Peng, and Wei Wang. 2023. STAR: Boosting Low-Resource Event
Extraction by Structure-to-Text Data Generation with Large Language Models.
CoRR abs/2305.15090 (2023). arXiv:2305.15090 doi:10.48550/ARXIV.2305.15090
[158] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation.
In The Thirteenth International Conference on Learning Representations. https:
//openreview.net/forum?id=oFBu7qaZpS
[159] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
rewriting in retrieval-augmented large language models. In Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing. 5303–
5315.
[160] Yubo Ma, Yixin Cao, YongChing Hong, and Aixin Sun. 2023. Large language
model is not a good few-shot information extractor, but a good reranker for
hard samples! arXiv preprint arXiv:2303.08559 (2023).
[161] Antonio Mallia, Joel Mackenzie, Torsten Suel, and Nicola Tonellotto. 2022. Faster
learned sparse retrieval with guided traversal. In Proceedings of the 45th Inter-
national ACM SIGIR Conference on Research and Development in Information
Retrieval. 1901–1905.
[162] Priyanka Mandikal and Raymond Mooney. 2024.
Sparse meets dense: A
hybrid approach to enhance scientific document retrieval.
arXiv preprint
arXiv:2401.04055 (2024).
[163] Dheeraj Mekala and Jingbo Shang. 2020. Contextualized Weak Supervision for
Text Classification. In ACL. 323–333.
[164] Yu Meng, Jiaxin Huang, Guangyuan Wang, Zihan Wang, Chao Zhang, Yu Zhang,
and Jiawei Han. 2020. Discriminative Topic Mining via Category-Name Guided
Text Embedding. In WWW’20.
[165] Yu Meng, Jiaming Shen, Chao Zhang, and Jiawei Han. 2018. Weakly-Supervised
Neural Text Classification. In Proceedings of the 27th ACM International Confer-
ence on Information and Knowledge Management. ACM, 983–992.
[166] Yu Meng, Jiaming Shen, Chao Zhang, and Jiawei Han. 2019. Weakly-supervised
hierarchical text classification. In AAAI.
[167] Yu Meng, Yunyi Zhang, Jiaxin Huang, Xuan Wang, Yu Zhang, Heng Ji, and
Jiawei Han. 2021. Distantly-Supervised Named Entity Recognition with Noise-
Robust Learning and Language Model Augmented Self-Training. In EMNLP.
10367–10378.
[168] Yu Meng, Yunyi Zhang, Jiaxin Huang, Chenyan Xiong, Heng Ji, Chao Zhang,
and Jiawei Han. 2020. Text Classification Using Label Names Only: A Language
Model Self-Training Approach. In EMNLP. 9006–9017.
[169] Yu Meng, Yunyi Zhang, Jiaxin Huang, Yu Zhang, and Jiawei Han. 2022. Topic
Discovery via Latent Space Clustering of Pretrained Language Model Represen-
tations. In WWW’22. 3143–3152.
[170] Yu Meng, Yunyi Zhang, Jiaxin Huang, Yu Zhang, Chao Zhang, and Jiawei Han.
2020. Hierarchical Topic Mining via Joint Spherical Tree and Text Embedding.
In KDD’20.
[171] Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram
Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu,
Asli Celikyilmaz, et al. 2023. Augmented language models: a survey. arXiv
preprint arXiv:2302.07842 (2023).
[172] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient esti-
mation of word representations in vector space. arXiv preprint arXiv:1301.3781
(2013).
[173] Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2021.
Noisy channel language model prompting for few-shot text classification. arXiv
preprint arXiv:2108.04106 (2021).
[174] Mandar Mitra and BB Chaudhuri. 2000. Information retrieval from documents:
A survey. Information retrieval 2 (2000), 141–163.
[175] Seyed Abbas Momtazi and Dietrich Klakow. 2010. Hierarchical Pitman–Yor
Language Model for Information Retrieval. In Proceedings of the 33rd Annual
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR 2010). 793–794.
[176] Eduardo Mosqueira-Rey, Elena Hernández-Pereira, David Alonso-Ríos, José
Bobes-Bascarán, and Ángel Fernández-Leal. 2023. Human-in-the-loop machine
learning: a state of the art. Artificial Intelligence Review 56, 4 (2023), 3005–3054.
[177] David Nadeau and Satoshi Sekine. 2007. A survey of named entity recognition
and classification. Lingvisticae Investigationes 30, 1 (2007), 3–26.
[178] Shahrzad Naseri, Jeffrey Dalton, Andrew Yates, and James Allan. 2021. Ceqe:
Contextualized embeddings for query expansion. In Advances in Information
Retrieval: 43rd European Conference on IR Research, ECIR 2021, Virtual Event,
March 28–April 1, 2021, Proceedings, Part I 43. Springer, 467–482.
[179] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan
Majumder, and Li Deng. 2016. Ms marco: A human-generated machine reading


A Survey on Retrieval And Structuring Augmented Generation with Large Language Models
KDD ’25, August 3–7, 2025, Toronto, ON, Canada
comprehension dataset. (2016).
[180] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith B Hall,
Daniel Cer, and Yinfei Yang. 2021. Sentence-t5: Scalable sentence encoders from
pre-trained text-to-text models. arXiv preprint arXiv:2108.08877 (2021).
[181] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego, Ji Ma,
Vincent Y Zhao, Yi Luan, Keith B Hall, Ming-Wei Chang, et al. 2021. Large dual
encoders are generalizable retrievers. arXiv preprint arXiv:2112.07899 (2021).
[182] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT.
arXiv preprint arXiv:1901.04085 (2019).
[183] Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. 2019. From doc2query to
docTTTTTquery. Online preprint 6, 2 (2019).
[184] Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document
expansion by query prediction. arXiv preprint arXiv:1904.08375 (2019).
[185] Zach Nussbaum, John X Morris, Brandon Duderstadt, and Andriy Mulyar. 2024.
Nomic embed: Training a reproducible long context text embedder. arXiv
preprint arXiv:2402.01613 (2024).
[186] Abiola Obamuyide and Andreas Vlachos. 2018. Zero-shot Relation Classification
as Textual Entailment. In Proceedings of the First Workshop on Fact Extraction
and VERification (FEVER). 72–78.
[187] Aytuğ Onan, Serdar Korukoğlu, and Hasan Bulut. 2016. Ensemble of keyword
extraction methods and classifiers in text classification. Expert Systems with
Applications 57 (2016), 232–247.
[188] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning
with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).
[189] OpenAI. 2024.
OpenAI o1 System Card.
CoRR abs/2412.16720 (2024).
arXiv:2412.16720 doi:10.48550/ARXIV.2412.16720
[190] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022.
Training language models to follow instructions with human feedback. Advances
in NeurIPS 35 (2022), 27730–27744.
[191] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright,
Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray,
John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens,
Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe.
2022. Training language models to follow instructions with human feedback. In
NeurIPS.
[192] Siru Ouyang, Jiaxin Huang, Pranav Pillai, Yunyi Zhang, Yu Zhang, and Jiawei
Han. 2023. Ontology Enrichment for Effective Fine-grained Entity Typing. arXiv
preprint arXiv:2310.07795 (2023).
[193] Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha. 2023. Fine-
tuning or retrieval? comparing knowledge injection in llms. arXiv preprint
arXiv:2312.05934 (2023).
[194] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion
Stoica, and Joseph E Gonzalez. 2023. MemGPT: Towards LLMs as Operating
Systems. arXiv preprint arXiv:2310.08560 (2023).
[195] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu.
2024. Unifying large language models and knowledge graphs: A roadmap. IEEE
Transactions on Knowledge and Data Engineering 36, 7 (2024), 3580–3599.
[196] Hao Peng, Jianxin Li, Yu He, Yaopeng Liu, Mengjiao Bao, Yangqiu Song, and
Qiang Yang. 2018. Large-Scale Hierarchical Text Classification with Recursively
Regularized Deep Graph-CNN. In WWW.
[197] Letian Peng, Zihan Wang, and Jingbo Shang. 2023. Less than One-shot: Named
Entity Recognition via Extremely Weak Supervision. In Findings of EMNLP.
13603–13616.
[198] Wenjun Peng, Guiyang Li, Yue Jiang, Zilong Wang, Dan Ou, Xiaoyi Zeng, Derong
Xu, Tong Xu, and Enhong Chen. 2024. Large language model based long-tail
query rewriting in taobao search. In Companion Proceedings of the ACM Web
Conference 2024. 20–28.
[199] José R Pérez-Agüera, Javier Arroyo, Jane Greenberg, Joaquin Perez Iglesias, and
Victor Fresno. 2010. Using BM25F for semantic search. In Proceedings of the 3rd
international semantic search workshop. 1–8.
[200] Gabriel Poesia, Oleksandr Polozov, Vu Le, Ashish Tiwari, Gustavo Soares,
Christopher Meek, and Sumit Gulwani. 2022. Synchromesh: Reliable code
generation from pre-trained language models. arXiv preprint arXiv:2201.11227
(2022).
[201] Jaroslav Pokorny. 2004. Web searching and information retrieval. Computing in
Science & Engineering 6, 4 (2004), 43–48.
[202] Meng Qu, Xiang Ren, Yu Zhang, and Jiawei Han. 2018. Weakly-supervised
Relation Extraction by Pattern-enhanced Embedding Learning. In WWW.
1257–1266.
[203] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxi-
ang Dong, Hua Wu, and Haifeng Wang. 2020. RocketQA: An optimized training
approach to dense passage retrieval for open-domain question answering. arXiv
preprint arXiv:2010.08191 (2020).
[204] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018.
Improving language understanding by generative pre-training. (2018).
[205] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano
Ermon, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language
Model is Secretly a Reward Model. In NeurIPS.
[206] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits
of transfer learning with a unified text-to-text transformer. Journal of machine
learning research 21, 140 (2020), 1–67.
[207] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.
Squad: 100,000+ questions for machine comprehension of text. arXiv preprint
arXiv:1606.05250 (2016).
[208] Alexander Ratner, Christopher De Sa, Sen Wu, Daniel Selsam, and Christopher
Ré. 2016. Data Programming: Creating Large Training Sets, Quickly. In NIPS.
[209] Wendi Ren, Yinghao Li, Hanting Su, David Kartchner, Cassie Mitchell, and
Chao Zhang. 2020. Denoising Multi-Source Weak Supervision for Neural Text
Classification. In EMNLP Findings.
[210] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance
framework: BM25 and beyond. Foundations and Trends® in Information Retrieval
3, 4 (2009), 333–389.
[211] Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu,
Mike Gatford, et al. 1995. Okapi at TREC-3. Nist Special Publication Sp 109
(1995), 109.
[212] Maya Rotmensch, Yoni Halpern, Abdulhakim Tlimat, Steven Horng, and David
Sontag. 2017. Learning a health knowledge graph from electronic medical
records. Scientific reports 7, 1 (2017), 5994.
[213] Dwaipayan Roy, Debjyoti Paul, Mandar Mitra, and Utpal Garain. 2016.
Using word embeddings for automatic query expansion.
arXiv preprint
arXiv:1606.07608 (2016).
[214] Oscar Sainz, Oier Lopez de Lacalle, Gorka Labaka, Ander Barrena, and Eneko
Agirre. 2021. Label Verbalization and Entailment for Effective Zero and Few-Shot
Relation Extraction. In EMNLP. 1199–1212.
[215] Gerard Salton and Christopher Buckley. 1988. Term-weighting approaches in
automatic text retrieval. Information processing & management 24, 5 (1988),
513–523.
[216] Diego Sanmartin. 2024. Kg-rag: Bridging the gap between knowledge and
creativity. arXiv preprint arXiv:2405.12035 (2024).
[217] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347
(2017).
[218] Jingbo Shang, Liyuan Liu, Xiang Ren, Xiaotao Gu, Teng Ren, and Jiawei Han.
2018. Learning Named Entity Tagger using Domain-Specific Dictionary. In
EMNLP.
[219] Jingbo Shang, Xinyang Zhang, Liyuan Liu, Sha Li, and Jiawei Han. 2020. NetTaxo:
Automated Topic Taxonomy Construction from Text-Rich Network. In WWW
(WWW ’20). 1908–1919.
[220] Yu-Ming Shang, Hongli Mao, Tian Tian, Heyan Huang, and Xian-Ling Mao. 2025.
From local to global: Leveraging document graph for named entity recognition.
Knowledge-Based Systems (2025), 113017.
[221] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu
Chen. 2023. Enhancing retrieval-augmented large language models with itera-
tive retrieval-generation synergy. arXiv preprint arXiv:2305.15294 (2023).
[222] Jiaming Shen, Wenda Qiu, Yu Meng, Jingbo Shang, Xiang Ren, and Jiawei Han.
2021. TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class
Names. In NAACL.
[223] Jiaming Shen, Zeqiu Wu, Dongming Lei, Jingbo Shang, Xiang Ren, and Jiawei
Han. 2017. Setexpan: Corpus-based set expansion via context feature selection
and rank ensemble. In ECML-PKDD’17. 288–304.
[224] Jiaming Shen, Zeqiu Wu, Dongming Lei, Chao Zhang, Xiang Ren, Michelle T
Vanni, Brian M Sadler, and Jiawei Han. 2018. Hiexpan: Task-guided taxonomy
construction by hierarchical tree expansion. In Proceedings of the 24th ACM
SIGKDD International Conference on Knowledge Discovery & Data Mining. 2180–
2189.
[225] Jiaming Shen, Yunyi Zhang, Heng Ji, and Jiawei Han. 2021. Corpus-based
Open-Domain Event Type Induction. In Proceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event /
Punta Cana, Dominican Republic, 7-11 November, 2021, Marie-Francine Moens,
Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for
Computational Linguistics, 5427–5440. doi:10.18653/V1/2021.EMNLP-MAIN.441
[226] Zhendong Shi, Jacky Keung, and Qinbao Song. 2014. An empirical study of bm25
and bm25f based feature location techniques. In Proceedings of the International
Workshop on Innovative Software Development Methodologies and Practices. 106–
114.
[227] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won
Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al.
2023. Large language models encode clinical knowledge. Nature 620, 7972
(2023), 172–180.
[228] Yangqiu Song and Dan Roth. 2014. On Dataless Hierarchical Text Classification.
In AAAI.
[229] Karen Sparck Jones. 1972. A statistical interpretation of term specificity and its
application in retrieval. Journal of documentation 28, 1 (1972), 11–21.


KDD ’25, August 3–7, 2025, Toronto, ON, Canada
Pengcheng Jiang et al.
[230] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph:
Deep and Responsible Reasoning of Large Language Model on Knowledge
Graph. In The Twelfth International Conference on Learning Representations.
https://openreview.net/forum?id=nnVO1PvbTv
[231] Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu,
Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten Rijke, and Zhaochun Ren.
2023. Learning to tokenize for generative retrieval. Advances in NeurIPS 36
(2023), 46345–46361.
[232] Xiaofei Sun, Xiaoya Li, Jiwei Li, Fei Wu, Shangwei Guo, Tianwei Zhang, and
Guoyin Wang. 2023. Text Classification via Large Language Models. In Findings
of EMNLP 2023. 8990–9005.
[233] Yueqing Sun, Qi Shi, Le Qi, and Yu Zhang. 2021. JointLK: Joint reasoning with
language models and knowledge graphs for commonsense question answering.
arXiv preprint arXiv:2112.02732 (2021).
[234] Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou. 2022.
Recitation-augmented language models. arXiv preprint arXiv:2210.01296 (2022).
[235] Ammar Tahir. 2023. Knowledge Graph GPT. https://github.com/iAmmarTahir/
KnowledgeGraphGPT.
[236] Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta,
Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a
differentiable search index. Advances in NeurIPS 35 (2022), 21831–21843.
[237] Rafael Teixeira de Lima, Shubham Gupta, Cesar Berrospi Ramis, Lokesh Mishra,
Michele Dolfi, Peter Staar, and Panagiotis Vagenas. 2025. Know Your RAG:
Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems. In
Proceedings of the 31st International Conference on Computational Linguistics:
Industry Track, Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-
Khalifa, Barbara Di Eugenio, Steven Schockaert, Kareem Darwish, and Apoorv
Agarwal (Eds.). Association for Computational Linguistics, Abu Dhabi, UAE,
39–57. https://aclanthology.org/2025.coling-industry.4/
[238] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of
information retrieval models. arXiv preprint arXiv:2104.08663 (2021).
[239] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi,
Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288 (2023).
[240] Andrew Trotman, Antti Puurula, and Blake Burgess. 2014. Improvements to
BM25 and language models examined. In Proceedings of the 19th Australasian
Document Computing Symposium. 58–65.
[241] Zhucheng Tu and Sarguna Janani Padmanabhan. 2022. MIA 2022 Shared Task
Submission: Leveraging Entity Representations, Dense-Sparse Hybrids, and
Fusion-in-Decoder for Cross-Lingual Question Answering.
arXiv preprint
arXiv:2207.01940 (2022).
[242] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you
need. Advances in NeurIPS 30 (2017).
[243] Somin Wadhwa, Silvio Amir, and Byron Wallace. 2023. Revisiting Relation
Extraction in the era of Large Language Models. In ACL. 15566–15589.
[244] Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying Liu, Haiyue Song, Jiwei Li, and
Sadao Kurohashi. 2023. GPT-RE: In-context Learning for Relation Extraction
using Large Language Models. In EMNLP. 3534–3547.
[245] Chengyu Wang, Jianing Wang, Minghui Qiu, Jun Huang, and Ming Gao. 2021.
Transprompt: Towards an automatic transferable prompting framework for
few-shot text classification. In Proceedings of the 2021 conference on empirical
methods in natural language processing. 2792–2802.
[246] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2022. Text embeddings by weakly-supervised
contrastive pre-training. arXiv preprint arXiv:2212.03533 (2022).
[247] Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang,
Jiwei Li, and Guoyin Wang. 2023. Gpt-ner: Named entity recognition via large
language models. arXiv preprint arXiv:2304.10428 (2023).
[248] Siyuan Wang, Zhongyu Wei, Jiarong Xu, Taishan Li, and Zhihao Fan. 2023. Uni-
fying Structure Reasoning and Language Pre-Training for Complex Reasoning
Tasks. IEEE/ACM Transactions on Audio, Speech, and Language Processing 32
(2023), 1586–1595.
[249] Shuohang Wang, Yichong Xu, Yuwei Fang, Yang Liu, Siqi Sun, Ruochen Xu,
Chenguang Zhu, and Michael Zeng. 2022. Training data is more valuable than
you think: A simple and effective method by retrieving from training data. arXiv
preprint arXiv:2203.08773 (2022).
[250] Xuan Wang, Yingjun Guan, Yu Zhang, Qi Li, and Jiawei Han. 2020. Pattern-
enhanced Named Entity Recognition with Distant Supervision. In 2020 IEEE
International Conference on Big Data (Big Data). 818–827.
[251] Xintao Wang, Qianwen Yang, Yongting Qiu, Jiaqing Liang, Qianyu He,
Zhouhong Gu, Yanghua Xiao, and Wei Wang. 2023. Knowledgpt: Enhanc-
ing large language models with retrieval and storage access on knowledge bases.
arXiv preprint arXiv:2308.11761 (2023).
[252] Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Qi Chen,
Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, et al. 2022. A neural
corpus indexer for document retrieval. Advances in NeurIPS 35 (2022), 25600–
25614.
[253] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amir-
reza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana
Arunkumar, David Stap, et al. 2022. Super-NaturalInstructions: Generalization
via Declarative Instructions on 1600+ NLP Tasks. In EMNLP. 5085–5109.
[254] Yue Wang, Dan Qiao, Juntao Li, Jinxiong Chang, Qishen Zhang, Zhongyi Liu,
Guannan Zhang, and Min Zhang. 2023. Towards Better Hierarchical Text
Classification with Data Generation. In Findings of ACL. 7722–7739.
[255] Zihan Wang, Dheeraj Mekala, and Jingbo Shang. 2021. X-Class: Text Classifica-
tion with Extremely Weak Supervision. In NAACL-HLT. 3043–3053.
[256] Zihan Wang, Peiyi Wang, Lianzhe Huang, Xin Sun, and Houfeng Wang. 2022.
Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach
for Hierarchical Text Classification. In ACL. 7109–7119.
[257] Jonatas Wehrmann, Ricardo Cerri, and Rodrigo C. Barros. 2018. Hierarchical
Multi-label Classification Networks. In ICML.
[258] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in NeurIPS 35 (2022), 24824–24837.
[259] Yilin Wen, Zifeng Wang, and Jimeng Sun. 2023. Mindmap: Knowledge graph
prompting sparks graph of thoughts in large language models. arXiv preprint
arXiv:2308.09729 (2023).
[260] Zhihao Wen and Yuan Fang. 2023. Augmenting low-resource text classification
with graph-grounded pre-training and prompting. In Proceedings of the 46th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 506–516.
[261] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina,
and Vicente Grau. 2024. Medical graph rag: Towards safe medical large language
model via graph retrieval-augmented generation. arXiv preprint arXiv:2408.04187
(2024).
[262] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis.
2023. Efficient streaming language models with attention sinks. arXiv preprint
arXiv:2309.17453 (2023).
[263] Jinfeng Xiao, Mohab Elkaref, Nathan Herr, Geeth De Mel, and Jiawei Han. 2023.
Taxonomy-Guided Fine-Grained Entity Set Expansion. In SDM’23.
[264] Yuxin Xiao, Zecheng Zhang, Yuning Mao, Carl Yang, and Jiawei Han. 2022. SAIS:
Supervising and Augmenting Intermediate Steps for Document-Level Relation
Extraction. In NAACL-HLT. 2395–2409.
[265] Yiqing Xie, Jiaming Shen, Sha Li, Yuning Mao, and Jiawei Han. 2022. Eider:
Empowering Document-level Relation Extraction with Efficient Evidence Ex-
traction and Inference-stage Fusion. In Findings of ACL. 257–268.
[266] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett,
Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor nega-
tive contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808
(2020).
[267] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. Recomp: Improving retrieval-
augmented lms with compression and selective augmentation. arXiv preprint
arXiv:2310.04408 (2023).
[268] Mingbin Xu and Hui Jiang. 2016. A FOFE-based local detection approach for
named entity recognition and mention detection. arXiv preprint arXiv:1611.00801
(2016).
[269] Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Desh-
pande, Xiaofeng Wang, and Zheng Li. 2024. Retrieval-augmented generation
with knowledge graphs for customer service question answering. In Proceedings
of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval. 2905–2909.
[270] Vikas Yadav and Steven Bethard. 2019. A survey on recent advances in named
entity recognition from deep learning models. arXiv preprint arXiv:1910.11470
(2019).
[271] Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu, and Zheng Zhang. 2021. A Unified
Generative Framework for Aspect-based Sentiment Analysis. In ACL-IJCNLP.
2416–2429.
[272] Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard
Hovy. 2016. Hierarchical Attention Networks for Document Classification.
In Proceedings of the 2016 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies, Kevin
Knight, Ani Nenkova, and Owen Rambow (Eds.). Association for Computational
Linguistics, San Diego, California, 1480–1489. doi:10.18653/v1/N16-1174
[273] Zeng Yang, Linhai Zhang, and Deyu Zhou. 2022. SEE-Few: Seed, Expand and
Entail for Few-shot Named Entity Recognition. In COLING. 2540–2550.
[274] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec,
Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2022. Retrieval-
augmented multimodal language modeling. arXiv preprint arXiv:2211.12561
(2022).
[275] Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christo-
pher D Manning, Percy S Liang, and Jure Leskovec. 2022. Deep bidirectional


A Survey on Retrieval And Structuring Augmented Generation with Large Language Models
KDD ’25, August 3–7, 2025, Toronto, ON, Canada
language-knowledge graph pretraining. Advances in NeurIPS 35 (2022), 37309–
37323.
[276] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure
Leskovec. 2021. QA-GNN: Reasoning with language models and knowledge
graphs for question answering. NAACL (2021).
[277] Fanghua Ye, Meng Fang, Shenghui Li, and Emine Yilmaz. 2023. Enhancing
conversational search: Large language model-aided informative query rewriting.
arXiv preprint arXiv:2310.09716 (2023).
[278] Sizhe Zhou Heng Ji Jiawei Ha Yizhu Jiao, Sha Li. 2024. Text2DB: Integration-
Aware Information Extraction with Large Language Model Agents. (2024).
[279] Changlong Yu, Weiqi Wang, Xin Liu, Jiaxin Bai, Yangqiu Song, Zheng Li, Yi-
fan Gao, Tianyu Cao, and Bing Yin. 2022. FolkScope: Intention knowledge
graph construction for e-commerce commonsense discovery. arXiv preprint
arXiv:2211.08316 (2022).
[280] Changlong Yu, Weiqi Wang, Xin Liu, Jiaxin Bai, Yangqiu Song, Zheng Li, Yifan
Gao, Tianyu Cao, and Bing Yin. 2023. FolkScope: Intention Knowledge Graph
Construction for E-commerce Commonsense Discovery. In Findings of ACL.
1173–1191.
[281] Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya
Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2022. Generate rather
than retrieve: Large language models are strong context generators. arXiv
preprint arXiv:2209.10063 (2022).
[282] Li Yujian and Liu Bo. 2007. A normalized Levenshtein distance metric. IEEE
transactions on pattern analysis and machine intelligence 29, 6 (2007), 1091–1095.
[283] Xiangrong Zeng, Daojian Zeng, Shizhu He, Kang Liu, and Jun Zhao. 2018. Ex-
tracting Relational Facts by an End-to-End Neural Model with Copy Mechanism.
In ACL. 506–514.
[284] ChengXiang Zhai and John Lafferty. 2001. A Study of Smoothing Methods for
Language Models Applied to Ad Hoc Information Retrieval. In Proceedings of the
24th Annual International ACM SIGIR Conference on Research and Development
in Information Retrieval (SIGIR 2001). 334–342.
[285] Chao Zhang, Fangbo Tao, Xiusi Chen, Jiaming Shen, Meng Jiang, Brian Sadler,
Michelle Vanni, and Jiawei Han. 2018. TaxoGen: Unsupervised Topic Taxonomy
Construction by Adaptive Term Embedding and Clustering. In KDD. 2701–2709.
[286] Haoyu Zhang, Jun Liu, Zhenhua Zhu, Shulin Zeng, Maojia Sheng, Tao Yang, Guo-
hao Dai, and Yu Wang. 2024. Efficient and Effective Retrieval of Dense-Sparse
Hybrid Vectors using Graph-based Approximate Nearest Neighbor Search. arXiv
preprint arXiv:2410.20381 (2024).
[287] Jian Zhang, Xu Wang, Hongyu Zhang, Hailong Sun, and Xudong Liu. 2020.
Retrieval-based neural source code summarization. In Proceedings of the
ACM/IEEE 42nd International Conference on Software Engineering. 1385–1397.
[288] Kai Zhang, Bernal Jimenez Gutierrez, and Yu Su. 2023. Aligning Instruction
Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors. In
Findings of ACL. 794–812.
[289] Lu Zhang, Jiandong Ding, Yi Xu, Yingyao Liu, and Shuigeng Zhou. 2021. Weakly-
supervised Text Classification Based on Keyword Graph. In EMNLP. 2803–2813.
[290] Peitian Zhang, Zheng Liu, Yujia Zhou, Zhicheng Dou, Fangchao Liu, and Zhao
Cao. 2024. Generative retrieval via term set generation. In Proceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 458–468.
[291] Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie.
2023. Retrieve anything to augment large language models. arXiv preprint
arXiv:2310.07554 (2023).
[292] Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. 2023. TableLlama: Towards
Open Large Generalist Models for Tables. arXiv preprint arXiv:2311.09206 (2023).
[293] Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang,
Christopher D Manning, and Jure Leskovec. 2022. Greaselm: Graph reasoning en-
hanced language models for question answering. arXiv preprint arXiv:2201.08860
(2022).
[294] Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional
networks for text classification. Advances in NeurIPS 28 (2015).
[295] Yu Zhang, Xiusi Chen, Yu Meng, and Jiawei Han. 2021. Hierarchical Metadata-
Aware Document Categorization under Weak Supervision. In WSDM’21. 770–
778.
[296] Yunyi Zhang, Minhao Jiang, Yu Meng, Yu Zhang, and Jiawei Han. 2023. PIEClass:
Weakly-Supervised Text Classification with Prompting and Noise-Robust Itera-
tive Ensemble Training. In EMNLP. 12655–12670.
[297] Yu Zhang, Yu Meng, Xuan Wang, Sheng Wang, and Jiawei Han. 2022. Seed-
Guided Topic Discovery with Out-of-Vocabulary Seeds. In NAACL’22. 279–290.
[298] Yunyi Zhang, Jiaming Shen, Jingbo Shang, and Jiawei Han. 2020. Empower
Entity Set Expansion via Language Model Probing. In ACL’20. 8151–8160.
[299] Yunyi Zhang, Ruozhen Yang, Xueqiang Xu, Rui Li, Jinfeng Xiao, Jiaming Shen,
and Jiawei Han. 2024. TELEClass: Taxonomy Enrichment and LLM-Enhanced
Hierarchical Text Classification with Minimal Supervision.
arXiv preprint
arXiv:2403.00165 (2024).
[300] Yu Zhang, Yunyi Zhang, Martin Michalski, Yucheng Jiang, Yu Meng, and Jiawei
Han. 2023. Effective Seed-Guided Topic Discovery by Integrating Multiple Types
of Contexts. In WSDM’23.
[301] Yu Zhang, Yunyi Zhang, Yanzhen Shen, Yu Deng, Lucian Popa, Larisa Shwartz,
ChengXiang Zhai, and Jiawei Han. 2024. Seed-Guided Fine-Grained Entity
Typing in Science and Engineering Domains. AAAI (2024).
[302] Zihan Zhang, Meng Fang, Ling Chen, and Mohammad-Reza Namazi-Rad. 2022.
Is Neural Topic Modelling Better than Clustering? An Empirical Study on
Clustering with Contextual Embeddings for Topics. In NAACL’22. 3886–3993.
[303] Siyun Zhao, Yuqing Yang, Zilong Wang, Zhiyuan He, Luna K Qiu, and Lili Qiu.
2024. Retrieval augmented generation (rag) and beyond: A comprehensive
survey on how to make your llms use external data more wisely. arXiv preprint
arXiv:2409.14924 (2024).
[304] Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. 2024. Dense text
retrieval based on pretrained language models: A survey. ACM Transactions on
Information Systems 42, 4 (2024), 1–60.
[305] Xuandong Zhao, Siqi Ouyang, Zhiguo Yu, Ming Wu, and Lei Li. 2023. Pre-trained
Language Models Can be Fully Zero-Shot Learners. In ACL. 15590–15606.
[306] Lingfeng Zhong, Jia Wu, Qian Li, Hao Peng, and Xindong Wu. 2023. A compre-
hensive survey on automatic knowledge graph construction. Comput. Surveys
56, 4 (2023), 1–62.
[307] Ben Zhou, Daniel Khashabi, Chen-Tse Tsai, and Dan Roth. 2018. Zero-Shot
Open Entity Typing as Type-Compatible Grounding. In EMNLP’18. 2065–2076.
[308] Sizhe Zhou, Suyu Ge, Jiaming Shen, and Jiawei Han. 2023. Corpus-Based
Relation Extraction by Identifying and Refining Relation Patterns. In ECML
PKDD. 20–38.
[309] Sizhe Zhou, Yu Meng, Bowen Jin, and Jiawei Han. 2024. Grasping the Essentials:
Tailoring Large Language Models for Zero-Shot Relation Extraction. arXiv
preprint arXiv:2402.11142 (2024).
[310] Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, Zhiyuan Liu, Weize Chen, Jianming
Wu, Jiesi Liu, Ruohang Feng, and Guoyang Zeng. 2024. D-Bot: Database Diag-
nosis System using Large Language Models. Proc. VLDB Endow. 17, 10 (2024),
2514–2527. doi:10.14778/3675034.3675043
[311] Xuanhe Zhou, Zhaoyan Sun, and Guoliang Li. 2024. Db-gpt: Large language
model meets database. Data Science and Engineering 9, 1 (2024), 102–111.
[312] Yuqi Zhu, Xiaohan Wang, Jing Chen, Shuofei Qiao, Yixin Ou, Yunzhi Yao, Shumin
Deng, Huajun Chen, and Ningyu Zhang. 2024. LLMs for knowledge graph
construction and reasoning: recent capabilities and future opportunities. World
Wide Web (WWW) 27, 5 (2024), 58. doi:10.1007/S11280-024-01297-W
[313] Yi Zhu, Ye Wang, Jipeng Qiang, and Xindong Wu. 2023. Prompt-learning for
short text classification. IEEE Transactions on Knowledge and Data Engineering
36, 10 (2023), 5328–5339.
[314] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chen-
long Deng, Haonan Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen. 2023.
Large language models for information retrieval: A survey. arXiv preprint
arXiv:2308.07107 (2023).
[315] Shengyao Zhuang, Bing Liu, Bevan Koopman, and Guido Zuccon. 2023. Open-
source large language models are strong zero-shot query likelihood models for
document ranking. arXiv preprint arXiv:2310.13243 (2023).
