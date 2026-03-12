# How to Solve Few-Shot Abusive Content Detection Using the Data We Actually Have

> Auto-converted plain text. Category: 04_Evaluation_Safety

How to Solve Few-Shot Abusive Content Detection
Using the Data We Actually Have
Viktor Hangya and Alexander Fraser
Center for Information and Language Processing, LMU Munich,
Munich Center for Machine Learning,
{hangyav, fraser}@cis.lmu.de
Abstract
Due to the broad range of social media platforms, the requirements of abusive language detection systems are varied
and ever-changing. Already a large set of annotated corpora with different properties and label sets were created,
such as hate or misogyny detection, but the form and targets of abusive speech are constantly evolving. Since, the
annotation of new corpora is expensive, in this work we leverage datasets we already have, covering a wide range of
tasks related to abusive language detection. Our goal is to build models cheaply for a new target label set and/or
language, using only a few training examples of the target domain. We propose a two-step approach: first we train
our model in a multitask fashion. We then carry out few-shot adaptation to the target requirements. Our experiments
show that using already existing datasets and only a few-shots of the target task the performance of models improve
both monolingually and across languages. Our analysis also shows that our models acquire a general understanding
of abusive language, since they improve the prediction of labels which are present only in the target dataset and can
benefit from knowledge about labels which are not directly used for the target task.
Keywords: abusive content detection, transfer learning, few-shot training
1.
Introduction
The wide spread of social media allowed us to com-
municate and share our opinions quickly and con-
veniently. However, it gives place to abusive con-
tent as well, which leaves some groups of people
vulnerable. To push back abusive online content,
various automated systems, and more importantly
datasets (Poletto et al., 2021), were introduced cov-
ering various text genres such as forum (de Gibert
et al., 2018), Twitter (Struß et al., 2019) or Insta-
gram posts (Suryawanshi et al., 2020) of various
languages (Vidgen and Derczynski, 2020), user
groups such as women (Fersini et al., 2018) or
LGBTQ+ (Leite et al., 2020) and tasks including
hate speech (de Gibert et al., 2018), offensive lan-
guage (Zampieri et al., 2019) or toxicity (Leite et al.,
2020) detection, etc.
On the other hand, there is constantly a need to
annotate new datasets supporting previously un-
seen target scenarios. To reduce annotation costs,
related work leveraged transfer learning to build sys-
tems across languages (Ranasinghe and Zampieri,
2020) and domains (Glavaš et al., 2020). But find-
ing the right source datasets is often challenging,
since the label sets could differ. To alleviate the
problem, previous work manually altered the label
sets of the source datasets in order to match them
to the target requirements. However, this approach
is problematic, because it requires expertise in abu-
sive language datasets, since i) the rules developed
by other researchers for manual label matching are
not reusable due to the rapid change in the ap-
plication requirements and ii) the definition of the
normal
abusive
hate
spam
external
 train
target
 task
offensive
disrespectful
fearful
untargeted
abusive
profanity
insult
normal
ME
Mt
few-shot target  train
abusive
profanity insult
normal
targeted
Figure 1: Two-step approach: ME is trained on the
(external) datasets we already have, followed by
its adaptation to the target task (Mt) with only a
few-shots. Labels not directly used for the target
task are underlined, target labels not contained in
the external datasets are bolded.
same label in some datasets could conflict, e.g.,
the offensive label of the OLID dataset includes
profane language (Zampieri et al., 2019), while the
same label does not in HASOC (Mandl et al., 2019).
Thus, a precise understanding of abusive language
phenomena is required. Additionally, iii) novel fine-
grained labels do not have alternatives to be trans-
ferred from. Such fine-grained labels, e.g., related
to a specific view of a given community regarding
arXiv:2305.14081v2  [cs.CL]  6 May 2024


an event, can be created on-the-fly as moderators
or affected people face them. Thus, our goal is to
eliminate the need for such rules and make infor-
mation transfer more flexible with minimal target
task annotations.
To this end, we introduce a method leveraging
multiple already existing (external) datasets in order
to understand general abusive language, allowing
to build models cheaply for the target requirements
without the need for manual dataset modifications.
As shown in Figure 1, different datasets can inform
the model about different types of abusive content.
Some classes can directly be leveraged for the tar-
get task due to their matching label names and
definitions. Others, such as labels with matching
names but conflicting definitions or labels which are
not contained in the target dataset at all, are lever-
aged only indirectly, by contributing to the general
abusive language awareness of the model. Our
approach consists of two steps: jointly training a
language model on multiple external datasets using
prompt-learning (Schick and Schütze, 2021). We
then adapt the resulting model to the target require-
ments in the second step, using only a few samples
per label from the target task (4-shots in the main
experiments), which could even be created on-the-
fly. Since the target task can contain unseen labels,
i.e., labels which are not contained in any of the
external datasets, or classes with conflicting label
definitions, at least a few annotated samples are
needed for model specialization.
We test our method on various tasks (e.g., hate,
abuse and misogyny detection or target identi-
fication) in both monolingual and cross-lingual
(English→German,
→Italian,
→Brazilian Por-
tuguese, and →Hindi) setups. Additionally, our
datasets cover multiple platforms including longer
forum posts and shorter Twitter messages. Our ex-
periments show improved performance when train-
ing using the external datasets compared to various
baselines, including both monolingual and cross-
lingual settings, on both binary and fine-grained
test sets. We find that even unseen target labels
are improved due to the better general abusive lan-
guage understanding of our models. Our ablation
study shows that external-only labels (labels which
do not occur in the label set of the task we are
carrying out) improve performance, showing that
they contribute to general understanding as well.
We experiment with different target data sizes and
find that although our approach is more beneficial
at lower sizes, when more data is available it is
also effective. Finally, we perform model diagnos-
tics using HateCheck (Röttger et al., 2021), further
supporting our claim of better general abusive lan-
guage understanding. Our contributions are the
following:1
1Our code is publicly available at:
• a multi-dataset training (MDT) approach, us-
ing prompt-learning based fine-tuning, for an
efficient few-shot training which supports the
ever-changing nature of abusive language de-
tection,
• applicability across languages and text genres
to support a wide range of target tasks cheaply,
• comprehensive analyses for a better under-
standing of model behavior.
2.
Related Work
To alleviate the issue of missing datasets for a
given target task, previous work leveraged trans-
fer learning techniques. Ranasinghe and Zampieri
(2020) built hate speech classifiers for Hindi, Span-
ish and Bengali by relying only on an English train-
ing dataset, while Glavaš et al. (2020) followed
a similar approach for cross-domain experiments.
They made the train and test corpora compatible us-
ing rule based label adaptation which, as discussed
above, is often difficult. In this work, we eliminate
this step and use external datasets without any
modifications. Furthermore, Wiegand et al. (2018)
showed that by adding seemingly similar English
samples to a small amount of German training data
the results decreased, while Nozza (2021) found
that in zero-shot cross-lingual models language
specific interjections are often misinterpreted lead-
ing to errors. These results indicate that select-
ing the right source dataset is not straight forward
(and perhaps impossible in some cases). In this
work, we leverage multiple external datasets for a
robust abusive language understanding, and use
a few-shots from the target dataset to specialize
our models to the target domain. Similarly, Röttger
et al. (2022) argues for the use of a small amount
of target language training samples in order to ex-
tend hate detection to multiple languages. However,
they focus on compatible binary datasets, while our
approach is compatible with fine-grained tasks with
unseen labels as well.
In contrast to transfer learning, the goal of multi-
task learning (MTL) is to build a shared model using
various tasks in order to improve the performance
on all of them, by exploiting common information
in some tasks, and to perform multiple tasks with
a single model (Caruana, 1997). Stickland and
Murray (2019) proposed an MTL method based on
pre-trained language models by introducing task
specific parameters in each layer. Due to nega-
tive task interference however, single task models
perform best in many cases. To mitigate the is-
sue of task interference, Pfeiffer et al. (2021) used
adapters (Houlsby et al., 2019) in a multitask set-
ting, showing that fusing information learned by
https://cistern.cis.lmu.de/multi_hs


task specific adapters can further boost the perfor-
mance on a target task. However, MTL is not able
to induce useful task dependencies when an imbal-
anced set, in terms of dataset sizes, is available.
In contrast, a set of auxiliary tasks were used to
improve the performance on a single target task in
(Watanabe et al., 2022). Similarly, Mehmood et al.
(2020) perform a final training step on the target
biomedical NER task after MTL, while Kapil and
Ekbal (2020) combine various abusive language
related datasets. However, these methods rely on
a large set of training data for the target task in
order to improve the performance, which is often
unavailable in the ever-changing field of abusive
language detection. Magnossão de Paula et al.
(2023) proposed to use task embeddings in order
to reduce negative transfer in MTL, but they only
consider binary abusive language detection tasks.
In comparison, we use a wide range of tasks with
heterogeneous label sets, including external-only
labels, when just a little target data is available, and
we show that this leads to a better general abusive
language understanding which positively impacts
even unseen target labels.
Our approach is also related to meta-learning
(Hospedales et al., 2021), where the goal is to
build a general model that is cheaply adaptable
to a target task that is unknown at the time of meta-
learning. In contrast, our goal is to build a classifier
for a resource poor task that is known when train-
ing the full model, by leveraging already existing
highly related datasets without strong expertise in
abusive language phenomena. Additionally, Wang
et al. (2021) showed that meta-learning has similar
performance to MTL, thus we only use MTL as a
baseline for simplicity.
3.
Approach
We consider two sets of training corpora in our multi-
dataset training (MDT) approach: external datasets
(DE = {Dei : i = 1..N}) which are not directly
related to the target task and the target dataset (Dt)
which is the target task for which we aim to build
a classifier. The former are off-the-shelf datasets
created for other tasks and/or languages containing
a few thousands or sometimes tens of thousands
of samples. In contrast, since our main goal is to
reduce the costs of building systems for novel target
tasks, Dt contains only a few samples, 4-shots per
label in our main experiments. MDT builds abusive
language classifiers in two steps (Figure 1). First,
we train a single model by fine-tuning a pre-trained
LM (M0) using only the external datasets in order
to learn general abusive language understanding
(resulting in ME), which we adapt to the specificities
of the target task in the second step (resulting in
Mt). In contrast to MTL or meta-learning, where
the final model supports multiple tasks, our final
models (Mt) are built for a single target task. This
imitates the use cases of social media platforms
which need to build a specialized model supporting
their own specific requirements. First, we discuss
the used prompt-learning technique (Schick and
Schütze, 2021), followed by the introduction of the
two-step training.
3.1.
Prompt-Learning
Prompt-learning was shown to be effective when
only a small training set is available. Instead of us-
ing classification heads on top of pre-trained LMs, it
aims to solve the task at hand using text generation.
Depending on the used LM architecture, various
techniques exist. We rely on encoder-only LMs in
our experiments, thus use the method proposed
by Schick and Schütze (2021), which employs the
masked language modeling task (MLM) to perform
text classification. Using pattern-verbalizer-pairs
(PVPs) an input sentence is first transformed using
the pattern, e.g., I’ll kill you. →I’ll kill you. It was
[MASK], and the task is to output the probability dis-
tribution over the vocabulary items at the masked
position. Finally, the verbalizer maps the highest
probability token, out of a set of valid tokens (see
below), to labels of a given dataset, e.g., threat-
ening →threat or neutral →normal. During
fine-tuning the model is trained to predict the token
associated with the correct input label, using the
MLM objective.
In our multi-dataset setup, we define PVPs for all
external datasets and the target dataset separately
(PV PE = {PV Pei : i = 1..N} and PV Pt). Having
a dedicated pattern and verbalizer for each dataset
makes our approach easy to be specialized for
each dataset, and at the same time easy to use,
since no single verbalizer handling all the labels is
required. We defined only two different patterns:
one for target detection datasets (X →X It was
targeted at [MASK], where X is the input text) and
another for the rest (X →X It was [MASK]). On the
other hand, the used verbalizers are specific for
the label set of each dataset, which can be defined
easily in general, i.e., we used 1-to-1 token to label
mapping in most cases. We refer to Table 6 of the
Appendix for more details about the exact patterns
and verbalizers for each used dataset.
3.2.
Multi-Dataset Training
Step 1: General Model Training (M0 →ME)
In
each step of the training process we randomly se-
lect an external dataset Dei and a batch of samples
from it. Other than the shared model core, i.e., the
pre-trained LM, we use the PVP related to Dei for
the forward-backward pass. For each dataset Dei
we use cross-entropy loss as the objective function


Lei to update the model. We run this process until
convergence.
Step 2: Model Specialization (ME →Mt)
In
order to adapt ME to the target task, we simply
continue training it on Dt by using the Dt specific
PVP. Similarly to the above, we use cross-entropy
loss Lt to update the model until convergence. As
shown by our experiments, the general abusive
language understanding learned by ME helps this
step to build a better model using just a few training
samples.
4.
Experimental Setup
4.1.
Datasets
We selected a wide range of datasets for our exper-
iments, covering various abusive language detec-
tion tasks, languages and text genres. We give a
short overview in the following and further details,
such as number of samples, exact PVPs, etc., in
Table 6 of the Appendix.
AMI
was created for the Evalita 2018 shared task
on Automatic Misogyny Identification (Fersini et al.,
2018), containing English and Italian tweets. We
use both the binary and fine-grained misogyny la-
bels as well as the target identification labels.
GermEval
was introduced for the shared task on
the Identification of Offensive Language in German
tweets (Struß et al., 2019). We used both binary
and fine-grained label sets.
HASOC
The shared task on Hate Speech and
Offensive Content Identification (Mandl et al., 2019)
introduces datasets for English, German and Hindi
containing Twitter and Facebook posts. We used its
fine-grained abuse and target identification labels.
HatEval
was built for SemEval 2019 Task 5 about
the detection of hate speech against immigrants
and women in Spanish and English Twitter mes-
sages (Basile et al., 2019). We used its binary hate
speech and target identification label sets.
LSA
is a large scale fine-grained abusive dataset
of English Tweets (Founta et al., 2018).
MLMA
Ousidhoum et al. (2019) introduced a mul-
tilingual and multi-aspect hate speech dataset of
English, French and Arabic Tweets. We leveraged
the fine-grained hostility labels in English.
Dataset
Labels
External datasets (DE)
AMI binary misogyny En
misogyny, normal
AMI fine-grained misogyny En
stereotype, dominance,
derailing,
sexual_harassment,
discredit
AMI binary target En
active, passive
HASOC fine-grained abusive En
hate, offensive, profanity
HASOC binary target En
targeted, untargeted
HatEval binary target En
individual, group
LSA fine-grained abusive En
abusive, hateful, spam,
normal
MLMA fine-grained hostility En
abusive, hateful, offensive,
disrespectful, fearful,
normal
SRW fine-grained abusive En
sexism, racism, normal
Target datasets (Dt)
HASOC fine-grained abusive En
hate, offensive, profanity
HASOC fine-grained abusive Hi
hate, offensive, profanity
HASOC fine-grained abusive De
hate, offensive, profanity
GermEval fine-grained offensive De
profanity, insult, abusive,
normal
ToLD-Br fine-grained toxicity Pt-Br
LGBTQ+phobia, obscene,
insult, racism, misogyny,
xenophobia, normal
OLID fine-grained target En
individual, group, other
Stormfront binary hate En
hate, normal
HatEval binary hate En
hateful, normal
HatEval binary hate Es
hateful, normal
OLID binary offensive En
offensive, normal
GermEval binary offensive De
offensive, normal
AMI binary misogyny En
misogyny, normal
AMI binary misogyny It
misogyny, normal
Table 1: Multi-dataset setup including external (DE)
and target (Dt) datasets. We consider similarly
defined but differently named labels to be the same,
such as hate and hateful, sexism and misogyny
or individual and active. We bold unseen labels
and underline labels which aren’t used in any target
dataset. We remove external datasets from DE
which are from the same source as a given target
dataset (in case of AMI, HASOC and HatEval).
OLID
The Offensive Language Identification
Dataset contains English tweets annotated with
offensive labels on three layers (Zampieri et al.,
2019). We used its binary offensive text and target
identification subsets.
SRW
is an English Twitter set created for sexism
and racism detection (Waseem and Hovy, 2016).
Stormfront
was created for hate speech detec-
tion containing English forum posts from the Storm-
front white supremacist forum (de Gibert et al.,
2018). It is annotated with binary labels.
ToLD-Br
is a Brazilian Portuguese Twitter dataset
annotated for toxicity detection (Leite et al., 2020).
We used its fine-grained label set containing a wide
range of labels, including misogyny.


4.2.
Multi-Dataset Setup
In the following we describe our multi-dataset setup,
i.e., the 9 corpora in the external set (DE) and the
13 target (Dt) datasets. For a high-level overview of
the setup, including labels of the external and target
datasets, we refer to Table 1. Note that although we
list multiple target datasets, we build a dedicated
model for each of them separately (Mt) in step 2
using ME which is trained on all the datasets in DE
jointly in step 1. Although the goal of MDT is to allow
for an easy external dataset selection specific to a
target task, we only consider a single external set
for all target datasets to save resources. However,
this general setup shows that an external set can
be selected without very careful data selection, and
gives a lower bound of the achievable performance
as we expect more benefits from a more specialized
setup.
The goal of the setup is to include a wide range
of datasets related to abusive language detec-
tion, such as hate speech, offense, abuse, sexism,
racism detection as well as target identification. Ad-
ditionally, we include datasets from the same task
category but with different label sets, e.g., HASOC
fine-grained abusive (hate, offensive, pro-
fane) and SRW fine-grained abusive (sex-
ism, racism, normal).
We only include English
datasets in DE, while we used both English and
non-English corpora (De, Hi, It, Pt-Br) as the tar-
get datasets to test cross-lingual transfer as well.
Furthermore, we test on Stormfront which contains
forum posts instead of Twitter and Facebook mes-
sages as the datasets in DE do. To avoid data leak-
age between the external train and the target test
sets (in case of AMI, HASOC and HatEval), i.e., to
filter samples which have the same input samples
but with different labels or inputs from different lan-
guages with the same labeling methodology, we
remove all datasets from DE which are from the
same authors as the test set, e.g., we omit all AMI
external datasets when training ME in step 1 in
case we test on AMI binary misogyny It.2
4.3.
Compared Systems
We compare MDT to four types of baseline systems.
We use off-the-shelf pre-trained LMs and train them
using the few-shot setup as in the second step of
our proposed approach without training them on
the external datasets (LM-base). As shown by Gu-
rurangan et al. (2020) fine-tuning LMs on the do-
main of the task of interest by further MLM training
on unlabeled data can improve down-stream task
performance. In order to test the effectiveness of
2Note that this filtering step requires us to train sepa-
rate ME models in these cases, resulting in 4 different
ME models overall.
this step in contrast to our approach which lever-
ages labels instead, we run MLM on the external
datasets of the above-mentioned setups for one
epoch (MLM). To test the importance of the two
separate steps of our approach we perform multi-
task learning, i.e., both the external datasets and
the target dataset are used in a single step similarly
as in step 1 in Section 3 (MTL). Additionally, we
test adapter-fusion (Pfeiffer et al., 2021) which elim-
inates negative task interference by first, training
independent adapters on each dataset, followed by
combining them for the target task (Fusion).3
Model parameters
We use xlm-roberta-base as
our base LM (Conneau et al., 2020) for all the base-
lines and our MDT setups as well. In our early ex-
periments we tested bert-base-multilingual-cased
(Devlin et al., 2019) as well, which resulted in simi-
lar conclusions. However, XLM-R benefited slightly
more from MDT which suggest that even larger
models might be able to exploit general informa-
tion from external datasets to a higher degree. For
evaluation we used macro averaged F1 score av-
eraged over 5 different seeds4 in order to reduce
the high variance issue of few-shot classification
(Zheng et al., 2022). We follow the standard n-shot
setting for few-shot learning. Due to the high label
bias of abusive language datasets, a large number
of samples have to be considered for annotation
in order to increase the training size of the minor-
ity classes with even a few examples. Thus, we
selected n = 4 as a reasonable trade-off between
costs and the amount of training data for the target
datasets (step 2). We experiment with different n
values shown in Figure 2. We used the full training
and validation sets of the external datasets (step
1). For all datasets we use the official train, vali-
dation and test splits if given, otherwise we take
80/20 train/test split of the full dataset and/or an
additional 80/20 split of the train set for final train-
ing and validation if the latter is not given. For the
implementation we used the Huggingface trans-
formers (Wolf et al., 2020) and OpenPrompt (Ding
et al., 2022) libraries for prompt-learning. The used
hyperparameters are: batch size 1,5 gradient accu-
mulation steps 16, warm-up steps 10, learning rate
5×10−5 and dropout 0.1. We ran a single epoch on
each external dataset in step 1, since we found that
longer training made our models biased towards
some of the most frequent labels. In contrast, we
used early stopping on the target n-shot validation
set in step 2.
3We use prompt-training in contrast to the original
work which uses classification heads.
4We use a single seed for MTL since it is trained on
a large set of inputs jointly.
5Due to limited GPU memory, we could not test on
larger batch sizes.


fine-grained
abusive
offensive
toxicity
target
HASOC (1/3)
GermEval
ToLD-Br
OLID
En
Hi
De
De (1/4)
Pt-Br (4/7)
En (1/3)
avg.
LM-base
32.76±5.94
33.59±5.68
32.55±11.71
21.48±2.22
8.25±3.54
36.94±3.80
27.59
MLM
35.98±7.70
35.81±3.79
28.70±9.94
21.97±3.89
8.98±2.76
40.88±5.38
28.72
MTL
13.22±0.00
16.38±0.00
24.10±0.00
19.84±0.00
10.69±0.00
14.11±0.00
16.39
Fusion
37.35±8.06
34.29±4.14
29.18±8.09
21.33±3.88
8.26±1.20
42.45±4.03
28.81
MDT
40.48±5.37
34.36±2.59
33.96±7.83
27.70±3.78
12.83±1.78
49.55±2.92
33.14
binary
hate
offensive
misogyny
Stormfront
HatEval
OLID
GermEval
AMI
En
En
Es
En
De
En
It
avg.
LM-base
52.82±6.36
57.31±4.98
45.60±6.78
52.70±7.18
50.89±5.33
57.31±4.25
60.33±10.45
53.85
MLM
54.88±4.89
58.91±2.53
46.37±9.00
53.63±7.09
52.81±1.16
48.68±8.20
64.24±5.82
54.22
MTL
54.22±0.00
54.40±0.00
48.85±0.00
56.00±0.00
47.05±0.00
49.73±0.00
45.32±0.00
50.80
Fusion
46.79±7.50
53.91±4.41
45.94±1.57
64.67±2.70
56.95±1.95
34.95±5.30
39.11±11.17
48.91
MDT
60.41±5.75
60.20±5.52
54.47±0.79
64.81±8.55
65.02±4.17
47.77±1.36
66.98±4.03
59.95
Table 2: Macro averaged F1 scores and standard deviation (%) on the fine-grained and binary target
datasets of our multi-dataset approach using 4-shot training. In case there are unseen labels in a given
target dataset, we highlight them together with the overall number of labels in parentheses. The best
result for each target dataset is in bold.
5.
Results
First, we present our main results followed by the
analysis of different few-shot sizes and the model
performance on each label separately. Then we
discuss an ablation study for a better understanding
of how the external datasets affect the final perfor-
mance. Finally, we briefly discuss our experiments
on HateCheck.
Our main results with 4-shot training are pre-
sented in Table 2. On a higher level it can be seen
that MDT improved over all baselines in 11 out of
13 cases (in 12 cases compared to LM-base). MDT
is significantly better (α = 0.05) than LM-base on all
datasets except on AMI En and HASOC De. We
used the significance test proposed in (Dror et al.,
2019), which accounts for the challenges of com-
paring deep neural networks, including the difficul-
ties due to the use of multiple random seeds. The
method compares the score distributions generated
by different runs (random seeds) of a given model
type using an approach based on Almost Stochas-
tic Dominance relation of the distributions. The
MLM, MTL and Fusion baselines also improve over
the LM-base system, however, not as consistently
and to a lesser extent than our approach. MTL
and Fusion even achieve lower performance than
LM-base when the averaged performance over all
datasets is considered. This indicates that i) relying
on the labels other than only the domain adaptation
effect of MLM is beneficial, ii) the two-step approach
of MDT is more effective, since the very low number
of samples of the target dataset are suppressed
by the external data samples when they are added
directly into MTL and iii) the small number of tar-
get dataset samples is not enough to properly fuse
dataset specific adapters (Fusion). Additionally,
training on the external datasets makes the models
more consistent over different runs, as shown by
the decreased standard deviation values on most
of the datasets.
MDT achieves comparable average improve-
ments on the fine-grained and the binary target
datasets. Looking at the former set, not only seen
but unseen labels as well were improved (even
on ToLD-Br with more than half of its labels un-
seen), suggesting that the general abusive lan-
guage aware ME model helps learning the fine-
grained label sets of these datasets even with only a
few-shots being available. We discuss the improve-
ments on the different labels in more details below
in the per label analysis section. The only excep-
tion is the HASOC fine-grained abusive Hi
dataset, where the MLM baseline achieved the best
results, although MDT also improved over LM-base.
Our conjecture is that this is partly due to the high
ratio of English content in the dataset caused by its
code-mixed nature, and as Table 2 suggests, MLM
tends to perform better on English target datasets
compared to non-English sets.
Although all labels of the binary target datasets
are seen, as mentioned the definitions of some
labels are different. For example, the offensive la-
bel of the OLID binary offensive En target
dataset includes profanity, while the same label
in the external HASOC fine-grained abusive
En dataset does not. However, due to the inclusion
of external training samples that are directly labeled
as profane, the model is trained on all the neces-
sary information. It only has to learn to combine


hate
offensive
profane
LM-base
30.64
24.39
43.24
MDT
52.94
26.64
41.85
(a) HASOC fine-grained abusive En
group
individual
other
LM-base
41.85
51.06
17.92
MDT
60.07
65.69
22.89
(b) OLID fine-grained target En
misogyny
racism
insult
xenophobia
LGBTQ+phobia
obscene
normal
LM-base
2.84
0.36
9.60
0.79
2.32
14.17
27.65
MDT
1.73
0.61
12.48
0.31
3.72
24.98
45.98
(c) ToLD-Br fine-grained toxic Pt-Br
hate
normal
LM-base
30.28
75.36
MDT
36.86
83.95
(d) Stormfront binary hate En
misogyny
normal
LM-base
55.26
65.40
MDT
69.33
64.64
(e) AMI binary misogyny It
Table 3: Per label 4-shot F1 scores (%). Unseen labels are bolded.
them in the final model of a given target dataset,
such as profanity and the more restrictive offensive
label of HASOC into the general offensive label of
OLID.
All the used external datasets are English. Com-
paring the improvements of the monolingual and
cross-lingual setups of MDT, i.e., English and non-
English target datasets, we found that the external
datasets are more beneficial monolingually. The
average improvements are 9.51% (ignoring AMI
misogyny En) and 6.09% respectively. This is
not surprising given that cross-lingual transfer learn-
ing is almost always less effective. Still, it shows
that the combination with English external datasets
is beneficial to non-English test corpora as well.
This is an important use case for reducing costs by
dramatically reducing the need for human annota-
tion. Additionally, all the external datasets contain
Twitter or Facebook posts, while the Stormfront tar-
get dataset contains forum posts which tend to be
longer and have different language use compared
to microblog posts. Even in this case, the improve-
ments of MDT are large compared with the baseline
showing the generality of our model.
N-shot analysis
Due to the high label bias of abu-
sive language datasets, we consider 4-shots of the
target datasets for efficiency. For example, out of
the 12 833 training samples in the ToLD-Br dataset
only 11 (0.08%) are labeled as racism, meaning
that a given annotator has to look at more than a
thousand text inputs to increase the number of this
minority label with just one sample. For a more com-
plete picture of the performance of our approach
however, we present experiments with different n
values on a few selected datasets in Figure 2, while
results on all the datasets are in Table 7 in the
appendix. Similarly to Zhao et al. (2021) we find
that although we average our results over 5 seeds,
1
4
8
16
32
64
20
40
60
80
n-shots
HASOC fine-grained abusive En
GermEval fine-grained offensive De
ToLD-Br fine-grained toxicity Pt-Br
Stormfront binary hate En
AMI binary misogyny It
LM-base
MDT
Figure 2: F1 scores (%) at different number (n-
shots) of target dataset samples comparing LM-
base with our MDT approach. Some datasets have
less than 64 samples available for the minority label.
the performance can be unstable at lower n val-
ues. It even decreases with the increase of training
samples in same cases. However, MDT steadily
outperforms LM-base, the gap only decreases at
higher n values. Still, even at n = 64 the baseline
performs worse on 2 of the 3 datasets. In contrast,
MDT has the largest improvements compared to
the baseline at lower n values, which shows the
strong advantage of using external datasets, es-
pecially for target datasets such as ToLD-Br, for


which acquiring even one racism sample is expen-
sive.
Per label analysis
We present per label F1
scores on a few selected datasets in Table 3. In
case of the fine-grained HASOC dataset in subtable
(a), the label hate was significantly improved, while
the performance on offensive and profane were
improved and decreased respectively with a simi-
lar margin. All labels were improved on the OLID
dataset in subtable (b), even the unseen other label
by almost 5 percentage points. Most interestingly,
3 out 4 unseen labels (5 out of 7 overall) were im-
proved on the ToLD-Br dataset in subtable (c).
Our conjecture is that the unseen insult label is
related to the fearful label of the external MLMA
dataset, since texts causing fear often involve in-
sults as well, which leads to this improvement. Addi-
tionally, as stated by the authors of ToLD-Br (Leite
et al., 2020), the unseen insult and obscene labels
were often confused by the annotators, indicating
their similarity, thus the latter could have also ben-
efited from the fearful instances of MLMA. Similarly,
the LMBTQ+phobia label is to some extent related
to sexism external instances, thus MDT can lever-
age their similarity automatically without the need
for manual label modifications. In contrast, xeno-
phobia, which is somewhat related to racism, was
not improved. However, since both models achieve
less than 1 percent F1, we believe that xenophobia
is simply too hard to classify, which is the reason
for no improvements in MDT. On the binary tar-
get datasets in subtables (d) and (e) the results
are similar as in case of the fine-grained datasets,
however all labels were improved in Stormfront,
while only misogyny improved in AMI.
Ablation study
We were interested in the addi-
tional value of labels which are not directly used in
the target datasets. Thus, we removed the external-
only labels from the external datasets in step 1,
i.e., all labels which are not part of a given target
dataset, and performed step 2 as normal (MDTabl).
We present experiments on datasets where MDT
outperformed LM-base in Table 4. In the major-
ity of the cases, especially for the binary datasets,
we found that removing labels from the external
datasets deteriorates the model’s performance sup-
porting our claim that by training first on a large set
of diverse datasets the model can learn a general
knowledge of abusive language which is then ben-
eficial for the target task. Even though some labels
in the external datasets aren’t directly used for a
given target task they are still beneficial.
HateCheck
is a test suite containing 29 func-
tional tests grouped into 11 classes (Röttger et al.,
2021). It focuses on testing various aspects of
MDT
MDTabl
fine-grained
HASOC abusive En
40.48
36.96
HASOC abusive Hi
34.36
38.38
HASOC abusive De
33.96
32.55
GermEval offensive De
27.70
28.04
ToLD-Br toxicity Pt-Br
12.83
8.20
OLID target EN
49.55
44.73
binary
Stormfront hate En
60.41
53.43
HatEval hate En
60.20
56.91
HatEval hate Es
54.47
52.69
OLID offensive En
64.81
45.09
GermEval offensive De
65.02
56.95
AMI misogyny It
66.98
65.45
avg.
47.56
43.28
Table 4: Macro averaged 4-shot F1 scores of our
ablation study which have labels removed from the
external datasets that are not needed in the target
datasets (MDTabl).
LM-base
MDT
F1-4
45.23
61.95
F5-6
48.59
49.72
F7-9
28.88
30.36
F10-11
50.00
52.38
F12-13
74.91
74.73
F14-15
29.15
24.89
F16-17
73.18
49.53
F18-19
27.14
9.57
F20-21
8.30
4.44
F22-24
14.51
21.82
F25-29
41.46
59.44
Table 5: Macro averaged 4-shot F1 scores on the
HateCheck functional test classes using HatEval
binary hate En as the target dataset. The re-
sults are averaged over test cases in the classes.
The naming FX −Y refers to the test class con-
taining test cases between X and Y .
hate speech detection models. It defines the fol-
lowing test classes: derogation (F1-4), threatening
language (F5-6), slur usage (F7-9), profanity us-
age (F10-11), profane reference (F12-13), negation
(F14-15), paraphrasing (F16-17), non-hate when
mentioning protected groups (F18-19), counter
speech (F20-21), abuse against non-protected tar-
gets (F22-24) and spelling variations (F25-29). We
evaluated MDT using HatEval binary hate
En as the target dataset compared to LM-base in
Table 5. We found that MDT improves on cases
testing a higher level of hate speech understand-
ing, such as derogation and threatening, or slur and
profanity usage, while the performance decreases
on cases testing linguistics phenomena, such as
negation or paraphrasing. These results further
support our claim that MDT results in models with
better abusive language understanding. On the
other hand, they also highlight sensitivity to linguis-


tic phenomena which should be improved in future
work.
6.
Conclusions
Due to the large variety of the abusive content to be
filtered, lack of resources is a major problem, larger
than for many other NLP tasks. In order to eliminate
the need for expensive dataset annotation for novel
application scenarios, and thus reduce costs, we
proposed a two-step multi-dataset approach (MDT)
which exploits datasets we already have to learn
general abusive language understanding and re-
quires only a few annotated samples for the target
task. Our experiments on various datasets showed
that external datasets improve few-shot classifica-
tion across tasks, text genres and languages. Ad-
ditionally, our analysis reveals that external-only
labels also contain useful information and conse-
quently, that unseen labels can be improved as
well, arguing for the utility of our approach.
Limitations
The aim of our approach is that given the require-
ments (labels and their definitions) of a target task,
train an abusive language classifier using external
datasets. Although our approach supports select-
ing more closely related datasets, e.g., only hate
speech or only misogyny and sexism datasets, we
used a broad range of abusive language related
external datasets in our experiments for simplicity,
and to save computational resources. On the one
hand, this setup shows the general applicability of
MDT. However, our results do not show the poten-
tial of more specialized (but still easy to set up)
configurations.
Additionally, we only used English corpora as
external datasets. Although English has the most
abusive language resources, datasets of other lan-
guages could also be used for this purpose. Our
results show that MDT is more beneficial monolin-
gually than cross-lingually, thus using resources
from the same language as a non-English target
dataset could be beneficial, which we did not test
in this work.
Finally, our analysis on HateCheck reveals some
weaknesses of our approach which indicate future
directions in combining abusive language datasets.
Acknowledgements
We thank the anonymous reviewers for their help-
ful feedback. The work was funded by the Euro-
pean Research Council (ERC; grant agreement
No. 640550) and by the German Research Foun-
dation (DFG; grant FR 2829/4-1 and FR 2829/7-1).
7.
Bibliographical References
Valerio Basile, Cristina Bosco, Elisabetta Fersini,
Debora Nozza, Viviana Patti, Francisco Manuel
Rangel Pardo, Paolo Rosso, and Manuela San-
guinetti. 2019. SemEval-2019 task 5: Multilingual
detection of hate speech against immigrants and
women in Twitter. In Proceedings of the 13th
International Workshop on Semantic Evaluation,
pages 54–63, Minneapolis, Minnesota, USA. As-
sociation for Computational Linguistics.
Rich Caruana. 1997. Multitask learning. Machine
learning, 28(1):41–75.
Alexis Conneau, Kartikay Khandelwal, Naman
Goyal, Vishrav Chaudhary, Guillaume Wenzek,
Francisco Guzmán, Edouard Grave, Myle Ott,
Luke Zettlemoyer, and Veselin Stoyanov. 2020.
Unsupervised cross-lingual representation learn-
ing at scale. In Proceedings of the 58th Annual
Meeting of the Association for Computational Lin-
guistics, pages 8440–8451, Online. Association
for Computational Linguistics.
Ona de Gibert, Naiara Perez, Aitor García-Pablos,
and Montse Cuadros. 2018. Hate speech dataset
from a white supremacy forum. In Proceedings
of the 2nd Workshop on Abusive Language On-
line (ALW2), pages 11–20, Brussels, Belgium.
Association for Computational Linguistics.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training
of deep bidirectional transformers for language
understanding. In Proceedings of the 2019 Con-
ference of the North American Chapter of the As-
sociation for Computational Linguistics: Human
Language Technologies, Volume 1 (Long and
Short Papers), pages 4171–4186, Minneapolis,
Minnesota. Association for Computational Lin-
guistics.
Ning Ding, Shengding Hu, Weilin Zhao, Yulin Chen,
Zhiyuan Liu, Haitao Zheng, and Maosong Sun.
2022.
OpenPrompt: An open-source frame-
work for prompt-learning. In Proceedings of the
60th Annual Meeting of the Association for Com-
putational Linguistics: System Demonstrations,
pages 105–113, Dublin, Ireland. Association for
Computational Linguistics.
Rotem Dror, Segev Shlomov, and Roi Reichart.
2019. Deep dominance - how to properly com-
pare deep neural models.
In Proceedings of
the 57th Annual Meeting of the Association for
Computational Linguistics, pages 2773–2785,
Florence, Italy. Association for Computational
Linguistics.


Elisabetta Fersini, Debora Nozza, and Paolo Rosso.
2018. Overview of the Evalita 2018 Task on Auto-
matic Misogyny Identification (AMI). Proceedings
of Sixth Evaluation Campaign of Natural Lan-
guage Processing and Speech Tools for Italian.
Final Workshop (EVALITA 2018), 12:59–66.
Antigoni Maria Founta, Constantinos Djouvas, De-
spoina Chatzakou, Ilias Leontia dis, Jeremy
Blackburn, Gianluca Stringhini, Athena Vakali,
Mic hael Sirivianos, and Nicolas Kourtellis. 2018.
Large Scale Crowdsourcing and Characteriza-
tion of Twitter Abusive Behavior. In Proceedings
of the Twelfth International AAAI Conference on
Web and Social Media, pages 491–500.
Goran Glavaš, Mladen Karan, and Ivan Vulić. 2020.
XHate-999: Analyzing and Detecting Abusive
Language Across Domains and Languages. In
Proceedings of the 28th International Conference
on Computational Linguistics, pages 6350–6365.
Suchin Gururangan, Ana Marasović, Swabha
Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey,
and Noah A. Smith. 2020. Don’t stop pretraining:
Adapt language models to domains and tasks.
In Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics, pages
8342–8360, Online. Association for Computa-
tional Linguistics.
Timothy Hospedales, Antreas Antoniou, Paul Mi-
caelli, and Amos Storkey. 2021. Meta-learning
in neural networks: A survey. IEEE transactions
on pattern analysis and machine intelligence,
44(9):5149–5169.
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski,
Bruna Morrone, Quentin De Laroussilhe, Andrea
Gesmundo, Mona Attariyan, and Sylvain Gelly.
2019. Parameter-Efficient Transfer Learning for
NLP. In Proceedings of the 36th International
Conference on Machine Learning, pages 2790–
2799.
Prashant Kapil and Asif Ekbal. 2020. A deep neural
network based multi-task learning approach to
hate speech detection. Knowledge-Based Sys-
tems, 210(October):106458.
João Augusto Leite, Diego Silva, Kalina Bontcheva,
and Carolina Scarton. 2020. Toxic language de-
tection in social media for Brazilian Portuguese:
New dataset and multilingual analysis. In Pro-
ceedings of the 1st Conference of the Asia-
Pacific Chapter of the Association for Computa-
tional Linguistics and the 10th International Joint
Conference on Natural Language Processing,
pages 914–924, Suzhou, China. Association for
Computational Linguistics.
Angel Felipe Magnossão de Paula, Paolo Rosso,
and Damiano Spina. 2023. Mitigating negative
transfer with task awareness for sexism, hate
speech, and toxic language detection. In 2023 In-
ternational Joint Conference on Neural Networks
(IJCNN).
Thomas Mandl, Sandip Modha, Prasenjit Ma-
jumder, Daksh Patel, M ohana Dave, Chintak
Mandlia, and Aditya Patel. 2019. Overview of the
HASOC track at FIRE 2019: Hate speech and
offensive content identification in Indo-European
languages. In 11th meeting of the Forum for
Information Retrieval Evaluation, pages 14–17.
Tahir Mehmood, Alfonso E. Gerevini, Alberto
Lavelli, and Ivan Serina. 2020. Combining multi-
task learning with transfer learning for biomedical
named entity recognition. Procedia Computer
Science, 176:848–857.
Debora Nozza. 2021. Exposing the limits of zero-
shot cross-lingual hate speech detection. In Pro-
ceedings of the 59th Annual Meeting of the As-
sociation for Computational Linguistics and the
11th International Joint Conference on Natural
Language Processing (Volume 2: Short Papers),
pages 907–914, Online. Association for Compu-
tational Linguistics.
Nedjma Ousidhoum, Zizheng Lin, Hongming
Zhang, Yangqiu Song, and Dit-Yan Yeung. 2019.
Multilingual and multi-aspect hate speech analy-
sis. In Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Process-
ing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP),
pages 4675–4684, Hong Kong, China. Associa-
tion for Computational Linguistics.
Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé,
Kyunghyun Cho, and Iryna Gurevych. 2021.
AdapterFusion: Non-destructive task composi-
tion for transfer learning. In Proceedings of the
16th Conference of the European Chapter of the
Association for Computational Linguistics: Main
Volume, pages 487–503, Online. Association for
Computational Linguistics.
Fabio Poletto, Valerio Basile, Manuela Sanguinetti,
Cristina Bosco, and Viviana Patti. 2021.
Re-
sources and benchmark corpora for hate speech
detection: a systematic review. Language Re-
sources and Evaluation, 55:477–523.
Tharindu Ranasinghe and Marcos Zampieri. 2020.
Multilingual offensive language identification with
cross-lingual embeddings. In Proceedings of the
2020 Conference on Empirical Methods in Natu-
ral Language Proce ssing, pages 5838–5844.


Paul Röttger, Debora Nozza, Federico Bianchi,
and Dirk Hovy. 2022. Data-efficient strategies
for expanding hate speech detection into under-
resourced languages.
In Proceedings of the
2022 Conference on Empirical Methods in Nat-
ural Language Processing, pages 5674–5691,
Abu Dhabi, United Arab Emirates. Association
for Computational Linguistics.
Paul Röttger, Bertie Vidgen, Dong Nguyen, Zeerak
Waseem, Helen Margetts, and Janet Pierrehum-
bert. 2021. HateCheck: Functional tests for hate
speech detection models. In Proceedings of the
59th Annual Meeting of the Association for Com-
putational Linguistics and the 11th International
Joint Conference on Natural Language Process-
ing (Volume 1: Long Papers), pages 41–58, On-
line. Association for Computational Linguistics.
Timo Schick and Hinrich Schütze. 2021. Exploiting
cloze-questions for few-shot text classification
and natural language inference. In Proceedings
of the 16th Conference of the European Chapter
of the Association for Computational Linguistics:
Main Volume, pages 255–269, Online. Associa-
tion for Computational Linguistics.
Asa Cooper Stickland and Iain Murray. 2019. BERT
and PALs: Projected Attention Layers for Effi-
cient Adaptation in Multi-Task Lea rning. In Pro-
ceedings of the 36th International Conference on
Machine Learning, pages 5986–5995.
Julia Maria Struß, Melanie Siegel, Josef Ruppen-
hofer, Michael Wiegand, and Manfred Klenner.
2019.
Overview of Germeval Task 2, 2019
Shared Task on the Identification of Offensive
Language. In Proceedings of the 15th Confer-
ence on Natural Language Processing, pages
354–365.
Shardul Suryawanshi, Bharathi Raja Chakravarthi,
Mihael Arcan, and Paul Buitelaar. 2020. Multi-
modal meme dataset (MultiOFF) for identifying
offensive content in image and text. In Proceed-
ings of the Second Workshop on Trolling, Aggres-
sion and Cyberbullying, pages 32–41, Marseille,
France. European Language Resources Associ-
ation (ELRA).
Bertie Vidgen and Leon Derczynski. 2020. Direc-
tions in abusive language training data, a sys-
tematic review: Garbage in, garba ge out. PLoS
ONE, 15(12):1–32.
Haoxiang Wang, Han Zhao, and Bo Li. 2021. Bridg-
ing multi-task learning and meta-learning: To-
wards efficient training and effective adaptation.
In International Conference on Machine Learning,
pages 10991–11002.
Zeerak Waseem and Dirk Hovy. 2016. Hateful sym-
bols or hateful people? predictive features for
hate speech detection on Twitter. In Proceed-
ings of the NAACL Student Research Workshop,
pages 88–93, San Diego, California. Association
for Computational Linguistics.
Taiki Watanabe, Tomoya Ichikawa, Akihiro Tamura,
Tomoya Iwakura, Chunpeng Ma, and Tsuneo
Kato. 2022.
Auxiliary learning for named en-
tity recognition with multiple auxiliary biomed-
ical training data. In Proceedings of the 21st
Workshop on Biomedical Language Processing,
pages 130–139, Dublin, Ireland. Association for
Computational Linguistics.
Michael
Wiegand,
Anastasija
Amann,
Ta-
tiana Anikina, Aikaterini Azoidou, Anastasia
Borisenkov, Kirstin Kolmorgen, Insa Kröger, and
Chris tine Schäfer. 2018. Saarland University’s
Participation in the GermEval Task 2018 (UdSW)
– Examining Different Types of Classifiers
and Features.
In Proceedings of GermEval
2018, 14th Conference on Natural Language
Processing (KONVENS 2018), pages 21–26.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi,
Pierric Cistac, Tim Rault, Remi Louf, Morgan
Funtowicz, Joe Davison, Sam Shleifer, Patrick
von Platen, Clara Ma, Yacine Jernite, Julien Plu,
Canwen Xu, Teven Le Scao, Sylvain Gugger,
Mariama Drame, Quentin Lhoest, and Alexan-
der Rush. 2020. Transformers: State-of-the-art
natural language processing. In Proceedings of
the 2020 Conference on Empirical Methods in
Natural Language Processing: System Demon-
strations, pages 38–45, Online. Association for
Computational Linguistics.
Marcos Zampieri, Shervin Malmasi, Preslav Nakov,
Sara Rosenthal, Noura Farra, and Ritesh Kumar.
2019. Predicting the type and target of offen-
sive posts in social media. In Proceedings of the
2019 Conference of the North American Chap-
ter of the Association for Computational Linguis-
tics: Human Language Technologies, Volume
1 (Long and Short Papers), pages 1415–1420,
Minneapolis, Minnesota. Association for Compu-
tational Linguistics.
Mengjie Zhao and Hinrich Schütze. 2021. Discrete
and soft prompting for multilingual models. In
Proceedings of the 2021 Conference on Empir-
ical Methods in Natural Language Processing,
pages 8547–8555, Online and Punta Cana, Do-
minican Republic. Association for Computational
Linguistics.
Mengjie Zhao, Yi Zhu, Ehsan Shareghi, Ivan
Vulić, Roi Reichart, Anna Korhonen, and Hin-


rich Schütze. 2021. A closer look at few-shot
crosslingual transfer: The choice of shots mat-
ters. In Proceedings of the 59th Annual Meeting
of the Association for Computational Linguistics
and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long
Papers), pages 5751–5767, Online. Association
for Computational Linguistics.
Yanan Zheng, Jing Zhou, Yujie Qian, Ming Ding,
Chonghua Liao, Li Jian, Ruslan Salakhutdinov,
Jie Tang, Sebastian Ruder, and Zhilin Yang. 2022.
FewNLU: Benchmarking state-of-the-art meth-
ods for few-shot natural language understanding.
In Proceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 501–516, Dublin,
Ireland. Association for Computational Linguis-
tics.
A.
Additional Details
We present details of the used datasets in Table 6,
such as source platform, number of samples (we
only used 4 samples per label for training in the
main experiments and an overall 16 samples fol-
lowing the original label distribution for validation
in case of the target datasets) and used PVPs. We
kept PVPs simple and uniform across datasets us-
ing English PVPs even for non-English datasets
as well. We note however that in our initial experi-
ments we tested machine translated PVPs which
did not lead to significantly different results (Zhao
and Schütze, 2021). Additionally, if a given token
related to a label is split by the tokenizer, e.g., dom-
inance →[domina, #nce], we take the averaged
probabilities of the subwords at the [MASK] posi-
tion as the probability of the related label. Finally,
we show complete results of all setups in Table 7
(containing each target dataset in a separate sub-
table stretching over multiple pages).


source
#train
#valid
#test
verbalizer
Pattern
AMI fine-grained misogyny En
Twitter
1,428
357
460
stereotypical →stereotype
dominance →dominance
derailing →derailing
harassment →sexual_harassment
discrediting →discredit
X →X It was [MASK]
AMI binary misogyny En
Twitter
3,200
800
1,000
sexist →misogyny
neutral →normal
X →X It was [MASK]
AMI binary misogyny It
Twitter
3,200
800
1,000
sexist →misogyny
neutral →normal
X →X It was [MASK]
GermEval fine-grained offensive De
Twitter
4,007
1,002
3,532
profane →profanity
insulting →insult
abusive →abusive
neutral →normal
X →X It was [MASK]
GermEval binary offensive De
Twitter
4,007
1,002
3,532
offensive →offensive
neutral →normal
X →X It was [MASK]
HASOC fine-grained abusive En
Twitter, Facebook
1,808
453
288
hate →hate
offensive →offensive
profane →profanity
X →X It was [MASK]
HASOC fine-grained abusive De
Twitter, Facebook
325
82
136
hate →hate
offensive →offensive
profane →profanity
X →X It was [MASK]
HASOC fine-grained abusive Hi
Twitter, Facebook
1,975
494
605
hate →hate
offensive →offensive
profane →profanity
X →X It was [MASK]
HatEval binary hate En
Twitter
3,055
764
850
hate →hateful
neutral →normal
X →X It was [MASK]
HatEval binary hate Es
Twitter
3,560
890
500
hate →hateful
neutral →normal
X →X It was [MASK]
LSA fine-grained abusive En
Twitter
29,728
7,433
9,291
abusive →abusive
hate →hateful
spam →spam
neutral →normal
X →X It was [MASK]
MLMA fine-grained hostility En
Twitter
5,549
1,388
1,735
abusive →abusive
hate →hateful
offensive →offensive
disrespectful →disrespectful
fearful →fearful
neutral →normal
X →X It was [MASK]
OLID binary offensive En
Twitter
10,592
2,648
860
offensive →offensive
neutral →normal
X →X It was [MASK]
SRW fine-grained abusive En
Twitter
6,504
1,626
2,033
sexist →sexism
racist →racism
neutral →normal
X →X It was [MASK]
Stormfront binary hate En
Stormfront forum
6,849
1,713
2,141
hate →hate
neutral →normal
X →X It was [MASK]
ToLD-Br fine-grained toxicity Pt-Br
Twitter
12,833
3,209
4,011
homophobic →LGBTQ+phobia
obscene →obscene
insulting →insult
racist →racism
sexist →misogyny
xenophobic →xenophobia
neutral →normal
X →X It was [MASK]
HASOC binary target En
Twitter, Facebook
4,681
1,171
1,153
targeted →targeted
general →untargeted
X →X It was [MASK]
AMI binary target En
Twitter
1,428
357
460
individual →active
group →passive
X →X It was targeted at [MASK]
HatEval binary target En
Twitter
3,732
933
1,318
individual →individual
group →group
X →X It was targeted at [MASK]
OLID fine-grained target En
Twitter
3,100
776
213
individual →individual
group →group
other →other
X →X It was targeted at [MASK]
Table 6: Dataset statistics for each (dataset, label configuration, language) triple. From left to right we
indicate the source platform of the dataset, the number of total train, validation and test samples, used
verbalizers (<predicted word> →<label>) which also indicates the labels of a given dataset, and
patterns (where X is the input sentence). We kept our PVPs simple, i.e., most labels are mapped 1-to-1 to
the same word, and we defined only two patterns. Note that we also used English PVPs for non-English
datasets, since it was shown to perform well (Zhao and Schütze, 2021). Since different datasets often
name the negative abuse class differently (e.g. no-hate, not-offensive, normal, etc.), we unified them by
using the frequent normal label name. Additionally, similarly defined but differently named labels, such as
hate and hateful or sexism and misogyny, are united by using the same verbalizers for them.


hate
offen.
profa.
avg.
1-shot
LM-base
20.37±16.46
34.26±2.39
25.90±12.28
26.85±3.80
MDT
31.80±21.12
30.37±10.10
34.41±20.77
32.19±2.87
4-shot
LM-base
30.64±17.57
24.39±8.99
43.24±5.11
32.76±5.94
MLM
38.35±21.08
31.91±5.89
37.67±11.63
35.98±7.70
MTL
0.00±0.00
39.66±0.00
0.00±0.00
13.22±0.00
Fusion
50.72±13.25
18.14±11.70
43.20±16.42
37.35±8.06
MDT
52.94±6.68
26.64±13.34
41.85±17.39
40.48±5.37
MDT-abl
49.94±10.26
19.01±5.60
41.93±13.72
36.96±2.77
8-shot
LM-base
36.09±18.50
28.12±8.97
49.68±5.98
37.96±5.32
MDT
54.44±8.98
23.55±13.52
42.44±8.61
40.14±3.48
16-shot
LM-base
52.37±12.34
26.66±5.91
54.62±7.43
44.55±3.80
MDT
62.35±5.45
23.21±5.76
56.37±8.98
47.31±2.23
32-shot
LM-base
60.08±11.25
18.43±12.46
54.77±9.13
44.43±3.22
MDT
61.53±8.39
18.26±14.98
55.28±6.11
45.03±2.20
64-shot
LM-base
39.86±23.03
30.31±5.62
53.82±9.36
41.33±10.86
MDT
60.07±4.17
25.59±7.61
41.23±23.92
42.29±10.14
(a) HASOC fine-grained abusive En
hate
offen.
profa.
avg.
1-shot
LM-base
42.46±22.52
46.03±20.92
10.56±6.93
33.02±12.44
MDT
24.35±18.43
65.74±10.60
18.16±5.98
36.08±7.51
4-shot
LM-base
28.81±22.59
47.96±24.69
20.88±4.24
32.55±11.71
MLM
29.85±18.60
36.99±21.05
19.26±5.83
28.70±9.94
MTL
0.00±0.00
72.30±0.00
0.00±0.00
24.10±0.00
Fusion
35.10±13.09
35.15±21.92
17.29±6.50
29.18±8.09
MDT
33.82±19.38
49.80±11.04
18.25±4.09
33.96±7.83
MDT-abl
28.59±16.25
47.73±27.24
21.33±13.17
32.55±11.16
8-shot
LM-base
16.45±14.06
48.32±18.31
18.15±5.24
27.64±4.79
MDT
36.94±14.65
42.17±18.52
20.55±5.22
33.22±6.65
16-shot
LM-base
26.37±22.50
62.63±10.75
16.98±5.52
35.33±8.64
MDT
43.96±12.16
60.61±9.98
23.63±6.81
42.73±7.51
32-shot
LM-base
5.95±7.35
54.02±8.79
20.67±9.13
26.88±2.44
MDT
34.71±16.89
49.94±17.30
15.99±10.30
33.55±7.46
64-shot
LM-base
25.98±16.26
60.51±9.13
19.28±7.88
35.26±5.96
MDT
27.59±12.55
54.48±12.94
23.66±2.62
35.24±5.26
(b) HASOC fine-grained abusive De
hate
offen.
profa.
avg.
1-shot
LM-base
29.00±15.58
24.21±16.02
27.77±13.25
26.99±2.91
MDT
33.87±13.10
29.49±16.19
27.90±22.33
30.42±5.26
4-shot
LM-base
25.57±18.46
33.84±11.71
41.37±12.38
33.59±5.68
MLM
23.09±16.30
34.63±14.80
49.71±5.74
35.81±3.79
MTL
0.00±0.00
49.13±0.00
0.00±0.00
16.38±0.00
Fusion
45.92±5.71
19.45±11.77
37.51±6.16
34.29±4.14
MDT
40.01±7.27
16.16±17.45
46.89±6.58
34.36±2.59
MDT-abl
44.52±6.07
18.81±15.40
51.82±8.43
38.38±5.45
8-shot
LM-base
40.28±14.55
31.14±12.85
42.40±14.03
37.94±2.53
MDT
47.25±3.31
19.72±8.11
44.61±8.93
37.19±1.72
(c) HASOC fine-grained abusive Hi
abus.
insult
profa.
normal
avg.
1-shot
LM-base
15.82±9.22
15.48±6.61
3.87±1.31
40.61±16.10
18.94±4.00
MDT
13.95±5.56
5.82±7.13
7.37±2.28
74.81±7.15
25.49±2.90
4-shot
LM-base
15.94±9.67
18.61±2.32
2.90±0.66
48.46±5.99
21.48±2.22
MLM
19.41±6.14
17.43±3.81
3.47±1.20
47.57±11.07
21.97±3.89
MTL
0.00±0.00
0.00±0.00
0.00±0.00
79.38±0.00
19.84±0.00
Fusion
12.24±4.51
13.35±5.24
4.10±0.79
55.63±17.20
21.33±3.88
MDT
23.30±9.48
14.53±4.17
8.60±2.00
64.35±7.62
27.70±3.78
MDT-abl
30.52±2.34
14.54±3.43
0.00±0.00
67.11±9.65
28.04±2.20
8-shot
LM-base
23.02±14.20
22.06±2.11
4.01±1.15
49.82±12.13
24.73±4.54
MDT
25.53±8.68
13.73±7.96
8.39±2.13
70.72±6.50
29.59±3.17
16-shot
LM-base
32.19±8.24
15.98±5.00
2.08±1.92
61.01±7.12
27.82±2.01
MDT
20.99±11.91
9.06±7.93
9.50±2.11
69.22±15.18
27.19±3.34
32-shot
LM-base
18.74±14.08
22.86±5.56
6.95±2.38
68.90±11.86
29.36±4.68
MDT
23.02±15.61
9.21±8.95
9.36±2.63
72.27±10.43
28.46±6.16
(d) GermEval fine-grained offensive De
misogyny
racism
insult
xenophobia
LGBTQ+hobia
obscene
normal
avg.
1-shot
LM-base
0.90±0.95
0.58±0.29
9.34±2.72
0.09±0.18
1.60±1.51
9.86±5.41
38.46±11.40
8.69±2.58
MDT
1.40±1.04
0.11±0.22
7.98±7.22
0.39±0.47
1.83±1.77
22.57±4.42
43.35±11.39
11.09±1.42
4-shot
LM-base
2.84±4.14
0.36±0.20
9.60±2.68
0.79±1.07
2.32±1.25
14.17±2.23
27.65±17.87
8.25±3.54
MLM
1.31±0.96
0.88±0.42
8.76±5.02
0.35±0.48
2.76±2.04
15.01±2.74
33.83±16.75
8.98±2.76
MTL
0.00±0.00
0.00±0.00
0.00±0.00
0.00±0.00
2.25±0.00
15.77±0.00
56.80±0.00
10.69±0.00
Fusion
0.75±0.41
0.31±0.10
8.71±2.59
0.07±0.13
2.73±0.82
7.74±3.00
37.48±2.10
8.26±1.20
MDT
1.73±0.82
0.61±0.89
12.48±6.33
0.31±0.38
3.72±3.62
24.98±4.50
45.98±6.12
12.83±1.78
MDT-abl
1.41±0.50
1.11±1.58
8.69±1.84
2.78±2.01
4.62±2.60
13.82±4.49
24.95±11.60
8.20±2.23
8-shot
LM-base
1.62±0.99
0.69±0.85
13.80±3.92
0.46±0.39
2.65±0.47
16.39±1.98
34.89±12.98
10.07±1.95
MDT
2.35±0.83
0.67±0.82
9.20±7.54
0.94±1.10
2.97±1.95
23.45±4.82
43.88±8.39
11.92±2.51
(e) ToLD-Br fine-grained toxic Pt-Br


group
indivi.
other
avg.
1-shot
LM-base
46.96±9.06
43.39±8.12
7.90±9.76
32.75±3.48
MDT
62.58±3.50
57.42±13.12
20.02±7.92
46.67±4.45
4-shot
LM-base
41.85±9.31
51.06±8.90
17.92±7.84
36.94±3.80
MLM
47.17±6.13
57.48±6.38
17.99±7.07
40.88±5.38
MTL
14.12±0.00
0.00±0.00
28.22±0.00
14.11±0.00
Fusion
58.21±3.19
51.48±10.50
17.66±8.32
42.45±4.03
MDT
60.07±11.50
65.69±8.59
22.89±6.60
49.55±2.92
MDT-abl
60.80±10.54
57.07±14.66
16.33±6.93
44.73±4.15
8-shot
LM-base
49.17±5.58
49.02±15.79
21.61±6.07
39.93±4.80
MDT
67.00±3.35
65.98±8.36
21.50±7.27
51.49±3.66
(f) OLID fine-grained target En
hate
normal
avg.
1-shot
LM-base
26.31±0.62
66.77±6.47
46.54±3.51
MDT
35.03±5.29
82.58±12.35
58.81±7.55
4-shot
LM-base
30.28±4.58
75.36±9.38
52.82±6.36
MLM
25.45±12.15
84.31±9.19
54.88±4.89
MTL
16.30±0.00
92.13±0.00
54.22±0.00
Fusion
30.18±3.51
63.40±11.55
46.79±7.50
MDT
36.86±2.87
83.95±9.03
60.41±5.75
MDT-abl
23.61±8.76
83.26±3.93
53.43±4.44
8-shot
LM-base
31.14±3.76
80.78±4.96
55.96±3.10
MDT
39.34±1.68
86.52±4.13
62.93±2.70
16-shot
LM-base
34.80±4.63
83.82±7.12
59.31±4.60
MDT
40.23±2.50
87.74±2.17
63.99±1.31
32-shot
LM-base
36.45±4.96
82.29±5.44
59.37±4.99
MDT
40.79±1.42
86.31±3.15
63.55±1.29
64-shot
LM-base
38.79±7.06
84.71±4.90
61.75±5.54
MDT
40.38±1.97
86.93±3.58
63.66±2.05
(g) Stormfront binary hate En
hateful
normal
avg.
1-shot
LM-base
46.18±13.74
62.61±8.12
54.40±5.75
MDT
46.49±11.06
54.76±7.88
50.62±3.70
4-shot
LM-base
55.20±3.04
59.42±10.52
57.31±4.98
MLM
55.08±5.70
62.73±8.40
58.91±2.53
MTL
38.72±0.00
70.09±0.00
54.40±0.00
Fusion
57.43±5.20
50.40±8.22
53.91±4.41
MDT
61.04±4.19
59.35±12.05
60.20±5.52
MDT-abl
56.18±4.34
57.63±8.86
56.91±2.82
8-shot
LM-base
48.29±12.01
63.44±11.82
55.86±5.42
MDT
61.87±1.41
65.62±7.88
63.75±3.83
16-shot
LM-base
53.38±10.13
64.45±3.70
58.91±4.10
MDT
61.95±4.45
65.02±6.73
63.48±3.62
32-shot
LM-base
62.72±3.13
65.49±6.95
64.10±2.48
MDT
64.30±5.29
64.97±6.95
64.63±3.71
64-shot
LM-base
61.90±6.97
70.57±4.09
66.23±2.27
MDT
63.73±4.54
69.86±4.22
66.79±2.40
(h) HatEval binary hate En
hateful
normal
avg.
1-shot
LM-base
44.34±14.02
46.96±22.46
45.65±7.07
MDT
46.49±11.06
54.76±7.88
50.62±3.70
4-shot
LM-base
51.81±6.45
39.39±19.59
45.60±6.78
MLM
42.27±12.12
50.46±25.12
46.37±9.00
MTL
28.19±0.00
69.52±0.00
48.85±0.00
Fusion
54.05±6.81
37.84±8.96
45.94±1.57
MDT
50.25±2.08
58.69±3.00
54.47±0.79
MDT-abl
44.02±2.61
61.36±3.56
52.69±0.96
8-shot
LM-base
40.61±6.04
63.78±2.79
52.20±3.70
MDT
54.02±3.96
59.23±3.47
56.62±0.87
(i) HatEval binary hate Es


offensive
normal
avg.
1-shot
LM-base
38.57±9.68
76.77±8.12
57.67±2.96
MDT
55.34±3.21
85.90±2.43
70.62±0.70
4-shot
LM-base
34.71±6.52
70.69±19.28
52.70±7.18
MLM
39.93±6.42
67.32±17.26
53.63±7.09
MTL
39.48±0.00
72.53±0.00
56.00±0.00
Fusion
54.17±2.20
75.17±5.15
64.67±2.70
MDT
55.55±3.70
74.06±14.25
64.81±8.55
MDT-abl
37.94±4.08
52.24±17.81
45.09±7.42
8-shot
LM-base
40.74±5.06
70.41±12.03
55.58±5.76
MDT
56.24±1.27
83.67±4.22
69.95±1.61
(j) OLID binary offensive En
offensive
normal
avg.
1-shot
LM-base
38.25±9.81
60.26±10.28
49.25±2.94
MDT
57.85±5.75
75.16±5.34
66.50±1.79
4-shot
LM-base
41.91±5.26
59.86±15.74
50.89±5.33
MLM
43.45±6.83
62.17±5.94
52.81±1.16
MTL
53.45±0.00
40.64±0.00
47.05±0.00
Fusion
50.96±3.03
62.95±6.31
56.95±1.95
MDT
55.74±3.17
74.31±9.90
65.02±4.17
MDT-abl
33.83±17.91
60.47±19.37
47.15±7.19
8-shot
LM-base
40.43±6.96
71.57±2.23
56.00±2.57
MDT
57.70±1.33
71.64±7.18
64.67±3.35
(k) GermEval binary offensive De
misogyny
normal
avg.
1-shot
LM-base
45.47±5.26
65.75±2.71
55.61±3.40
MDT
49.81±14.77
39.48±10.09
44.64±3.18
4-shot
LM-base
56.02±3.10
58.61±9.18
57.31±4.25
MLM
59.22±5.13
38.14±19.71
48.68±8.20
MTL
31.28±0.00
68.18±0.00
49.73±0.00
Fusion
42.23±26.40
27.68±33.90
34.95±5.30
MDT
53.57±5.14
41.97±5.13
47.77±1.36
MDT-abl
57.54±7.07
50.77±4.11
54.16±2.78
8-shot
LM-base
58.15±4.89
55.99±11.84
57.07±4.98
MDT
53.57±3.48
48.07±8.03
50.82±5.02
(l) AMI binary sexism En
misogyny
normal
avg.
1-shot
LM-base
56.37±17.70
59.66±11.53
58.02±8.47
MDT
53.10±8.67
45.78±13.44
49.44±4.03
4-shot
LM-base
55.26±21.34
65.40±4.06
60.33±10.45
MLM
69.23±5.45
59.25±10.89
64.24±5.82
MTL
40.79±0.00
49.86±0.00
45.32±0.00
Fusion
53.51±26.77
24.72±30.39
39.11±11.17
MDT
69.33±5.39
64.64±5.10
66.98±4.03
MDT-abl
66.54±7.19
64.35±5.03
65.45±4.02
8-shot
LM-base
70.77±3.42
60.21±5.12
65.49±3.17
MDT
68.95±8.80
65.65±6.41
67.30±6.46
16-shot
LM-base
74.11±2.37
70.00±10.28
72.05±5.73
MDT
75.82±3.34
71.74±3.27
73.78±2.98
32-shot
LM-base
77.83±6.63
73.35±5.90
75.59±6.12
MDT
78.33±1.54
73.77±4.60
76.05±2.52
64-shot
LM-base
81.23±1.86
76.76±3.94
79.00±2.84
MDT
80.40±1.22
74.81±4.19
77.61±2.65
(m) AMI binary misogyny It
Table 7: Per label and macro averaged F1 scores for each target dataset. LM-base, MLM, MTL and
Fusion rows indicate the model trained on the target task only, the masked language modeling, multitask
learning and adapter fusion baselines, MDT our proposed approach, while MDT-abl. refers to the ablation
studies where external only labels are removed from the external datasets.
