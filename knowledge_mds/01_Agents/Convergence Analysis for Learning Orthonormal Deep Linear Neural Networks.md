# Convergence Analysis for Learning Orthonormal Deep Linear Neural Networks

> Auto-converted plain text. Category: 01_Agents

1
Convergence Analysis for Learning Orthonormal
Deep Linear Neural Networks
Zhen Qin, Xuwei Tan, and Zhihui Zhu, Member, IEEE
Abstract—Enforcing orthonormal or isometric property for
the weight matrices has been shown to enhance the training of
deep neural networks by mitigating gradient exploding/vanishing
and increasing the robustness of the learned networks. However,
despite its practical performance, the theoretical analysis of
orthonormality in neural networks is still lacking; for example,
how orthonormality affects the convergence of the training process.
In this letter, we aim to bridge this gap by providing convergence
analysis for training orthonormal deep linear neural networks.
Specifically, we show that Riemannian gradient descent with an
appropriate initialization converges at a linear rate for training
orthonormal deep linear neural networks with a class of loss
functions. Unlike existing works that enforce orthonormal weight
matrices for all the layers, our approach excludes this requirement
for one layer, which is crucial to establish the convergence
guarantee. Our results shed light on how increasing the number
of hidden layers can impact the convergence speed. Experimental
results validate our theoretical analysis.
Index Terms—Deep neural networks, orthonormal structure,
convergence analysis, Riemannian optimization
I. INTRODUCTION
E
FORCING orthonormal or isometric properties of the
weight matrices has numerous advantages for the practice
of deep learning: (i) it provides a better initialization [1, 2],
(ii) it mitigates the problem of exploding/vanishing gradients
during training [3–7], (iii) the resulting orthonormal neural
networks [2, 8–16] exhibit improved robustness [17] and
reduced overfitting issues [18].
Various approaches have been proposed for training neural
networks, mainly falling into two categories: soft orthonormal-
ity and hard orthonormality. The first category of methods, such
as those in [10, 12, 17, 19], adds an additional orthonormality
regularization term to the training loss, resulting in weight
matrices that are approximately orthonormal. In contrast, the
other methods, as found in [9, 11, 20], learn weight matrices
that are exactly orthonormal through the use of Riemannian
optimization algorithms on the Stiefel manifold.
While orthonormal neural networks demonstrate strong
practical performance, there remains a gap in the theoretical
analysis of orthonormality in neural networks. For example,
convergence analysis for training neural networks has been
extensively studied [21–28]. However, all these results focus on
standard training without orthonormal constraints, making them
inapplicable to the training of orthonormal neural networks.
To the best of our knowledge, there is a lack of rigorous
Z. Qin, X. Tan and Z. Zhu are with the Department of Computer Science
and Engineering, the Ohio State University, OH 43210, USA (e-mail: {qin.660,
tan.1206, zhu.3440}@osu.edu).
This work was partly supported by NSF grants CCF-2240708 and CCF-
2241298.
convergence analysis even for orthonormal deep linear neural
networks (ODLNNs). Despite its linear structure, a deep linear
neural network still presents a non-convex training problem
and has served as a testbed for understanding deep neural
networks [21–24]. In this letter, we aim to understand the
effect of the orthonormal structure on the training process by
studying ODLNN.
Our contribution Specifically, we provide a local convergence
rate of Riemannian gradient descent (RGD) for training the
ODLNN. To achieve this, unlike existing works [9, 11, 17] that
impose orthonormal constraints on all the weight matrices, we
exclude such a constraint for one layer (say the weight matrix
in the first hidden layer). The exclusion of a specific layer plays
a crucial role in analyzing the convergence rate. Our findings
demonstrate that within a specific class of loss functions,
adhering to the restricted correlated gradient condition [29],
the RGD algorithm exhibits linear convergence speed when
appropriately initialized. Notably, our results also indicate that
as the number of layers in the network increases, the rate
of convergence only experiences a polynomial decrease. The
validity of our theoretical analysis has been confirmed by
experiments.
Notation: We use bold capital letters (e.g., A) to denote
matrices, bold lowercase letters (e.g., a) to denote vectors, and
italic letters (e.g., a) to denote scalar quantities. The superscript
(·)⊤denotes the transpose. ∥A∥and ∥A∥F respectively
represent the spectral norm and Frobenius norm of A. σmin(A)
is the smallest singular value of A. The condition number of A
is defined as κ(A) =
∥A∥
σmin(A). ∥a∥2 is the l2 norm of a. For a
positive integer K, [K] denotes the set {1, . . . , K}. b = Ω(a)
represents b ≥ca for some universal constant c.
II. RIEMANNIAN GRADIENT DESCENT FOR ORTHONORMAL
DEEP LINEAR NEURAL NETWORKS
Problem statement Given a training set {(xi, y⋆
i )}n
i=1 ∈
Rdx × Rdy, our goal is to estimate a hypothesis (predictor)
from a parametric family H := {hθ : Rdx →Rdy|θ ∈Θ} by
minimizing the following empirical risk:
min
θ∈Θ g(θ) = 1
n
n
X
i=1
l(hθ(xi); y⋆
i ),
(1)
where l(hθ(xi); y⋆
i ) is a suitable loss that captures the differ-
ence between the network prediction hθ(xi) and the label y⋆
i .
For convenience, we stack all the training samples together as
X =
x1
· · ·
xn

and Y ⋆=
y⋆
1
· · ·
y⋆
n

.
Our main focus is on orthonormal deep linear neural net-
works (ODLNNs), which are fully-connected neural networks
of form hθ(xi) = WN · · · W1xi with Wi ∈Rdi×di−1 for
i ∈[N], where d0 = dx and dN = dy. In ODLNNs, we
arXiv:2311.14658v2  [cs.LG]  29 Feb 2024


2
further assume that the weight matrices to be row orthogonal or
column orthogonal depending on the dimension. Without loss of
generality, we assume that all the matrices {Wi}i≥2 are column
orthonormal, except for W1. This is different to the previous
works [9, 11, 17] which impose orthonormal constraints on all
the weight matrices. Allowing W1 to be unstructured offers
more flexibility, as otherwise WN · · · W1 can only represent
an orthonormal matrix, which restricts the output the same
energy as input (i.e., ∥y⋆
i ∥2 = ∥xi∥2). The choice of free
weight matrix can vary, and the following analysis would still
hold. Now the training loss can be written as
min
Wi ∈Rdi×di−1
i ∈[N]
g(WN, . . . , W1) = L(WN · · · W1X; Y ⋆),
s. t.
W ⊤
i Wi = Idi−1,
i = 2, . . . N,
(2)
where L denotes a loss function encompassing all samples.
Definition 1 (Data model): Following the previous work
on deep linear neural networks [21, 22], we assume that the
dataset X is whitened, i.e., its empirical covariance matrix
is an identity matrix as XX⊤= Idx. Also assume that the
output is generated by a teacher ODLNN model1, i.e., Y ⋆=
W ⋆
N · · · W ⋆
1 X, where W ⋆
i
∈Rdi×di−1 and {W ⋆
i }i≥2 are
column orthonormal matrices.
Stiefel manifold The Stiefel manifold St(m, n) = {C ∈
Rm×n : C⊤C = In} is a Riemannian manifold that is
composed of all m × n orthonormal matrices. We can regard
St(m, n) as an embedded submanifold of a Euclidean space
and further define TCSt := {A ∈Rm×n : A⊤C+C⊤A = 0}
as its tangent space at the point C ∈St(m, n). For any
B ∈Rm×n, the projection of B onto TCSt is given by [30]
PTCSt(B) = B −1
2C(B⊤C + C⊤B),
(3)
and its orthogonal complement is P⊥
TCSt(B) = B−PTCSt(B)=
1
2C(B⊤C + C⊤B). When we have a gradient B defined in
the Hilbert space, we can use the projection operator (3) to
compute the Riemannian gradient PTCSt(B) on the tangent
space of the Stiefel manifold. To project b
C = C −cPTCSt(B)
with any positive constant c back onto the Stiefel manifold,
we can utilize the polar decomposition-based retraction, i.e.,
RetrC( b
C) = b
C( b
C⊤b
C)−1
2 .
(4)
Riemannian gradient descent (RGD) Given the gradient
∇Wig(W (t)
N , . . . , W (t)
1 ), we can compute the Riemannian
gradient PTWiSt
 ∇Wig(W (t)
N , . . . , W (t)
1 )

on the Stiefel man-
ifold via (3). To streamline the notation, let us represent
∇Wig(W (t)
N , . . . , W (t)
1 ) as ∇W (t)
i g. Now the weight matrices
can be updated via the following RGD:
W (t+1)
1
= W1
(t) −µγ∇W (t)
1 g,
W (t+1)
i
= RetrWi

W (t)
i
−µPTWiSt
 ∇W (t)
i g

, i ≥2,
(5)
where µ > 0 is the learning rate for {Wi} and γ > 0 controls
the ratio between the learning rates for W1 and {Wi}i≥2.
The discrepant learning rates in (5) are used to accelerate the
1Here, for the sake of simplifying the subsequent analysis, we designate
the teacher model as the ODLNN, which can encompass any linear model.
convergence rate of W1 since the energy of W ⋆
1 and {W ⋆
i }i≥2
are unbalanced, i.e., ∥W ⋆
1 ∥2 = ∥Y ⋆∥2 and ∥W ⋆
i ∥2 = 1 when
the dataset X is whitened, i.e. XX⊤= Idx.
III. CONVERGENCE ANALYSIS
In this section, we will delve into the convergence rate
analysis of RGD for training ODLNNs. Towards that goal, we
will use the teacher model introduced in Definition 1 that the
training samples (X, Y ⋆) are generated according to Y ⋆=
W ⋆
N · · · W ⋆
1 X. Given the nonlinear nature of the retraction
operation in the RGD, we will study the convergence in terms
of the weight matrices W = {Wi} and W⋆= {W ⋆
i }. But we
will show that the convergence can be equivalently established
in terms of the outputs.
Distance measure Consider that the factors in W⋆are iden-
tifiable up to orthonormal transforms since W ⋆
N · · · W ⋆
1 X =
W ⋆
NRN−1R⊤
N−1W ⋆
N−1RN−2 · · · W ⋆
1 X for any orthonormal
matrices Ri ∈Odi×di, i ∈[N −1]. Also, W ⋆
1 and W ⋆
i could
be imbalanced as W ⋆
i are orthonormal for i ≥2. This dis-
crepancy can be quantified by observing that ∥W ⋆
1 ∥= ∥Y ⋆∥
(since the input matrix X is whitened) and ∥W ⋆
i ∥= 1 for all
i ≥2. Thus, we propose the following measure to capture the
distance between two sets of factors:
dist2(W, W⋆)= min
Ri∈Odi×di ,
i∈[N−1]
N
X
i=2
∥Y ⋆∥2∥Wi −R⊤
i W ⋆
i Ri−1∥2
F
+∥W1 −R⊤
1 W ⋆
1 ∥2
F .
(6)
Here the coefficient ∥Y ⋆∥2 is to harmonize the energy levels
between W = {Wi}i≥2 and W1. The following result eluci-
dates the connection between dist2(W, W⋆) and ∥Y −Y ⋆∥2
F ,
guaranteeing the convergence of Y as W approaches the global
minima.
Lemma 1:
Assume a whitened input X
∈
Rdx×n,
i.e. XX⊤= Idx. Let Y
= WN · · · W1X and Y ⋆=
W ⋆
N · · · W ⋆
1 X, where Wi, W ⋆
i
∈Rdi×di−1 are orthonor-
mal for i = 2, . . . , N. Given that ∥W1∥2 ≤
9∥Y ⋆∥2
4
and
∥W ⋆
1 ∥2 = ∥Y ⋆∥2, we can get
∥Y −Y ⋆∥2
F ≥
1
(16N −8)κ2(Y ⋆)dist2(W, W⋆),
(7)
∥Y −Y ⋆∥2
F ≤9N
4 dist2(W, W⋆).
(8)
Proof
Using the result [29, eq. (E.4)], for any j ≥2, we
have that ∥WN · · · Wj −W ⋆
N · · · W ⋆
j Rj−1∥2
F ≤4∥Y −Y ⋆∥2
F
σ2
min(Y ⋆)
for any Ri ∈Odi×di. It follows that
∥WN−1 −R⊤
N−1W ⋆
N−1RN−2∥2
F
= ∥W ⋆
NRN−1WN−1 −W ⋆
NW ⋆
N−1RN−2∥2
F
≤2∥W ⋆
NRN−1 −WN∥2
F ∥WN−1∥2
+2∥WNWN−1 −W ⋆
NW ⋆
N−1RN−2∥2
F
≤16∥Y −Y ⋆∥2
F
σ2
min(Y ⋆)
.
(9)
Similarly, we get ∥Wi −R⊤
i W ⋆
i Ri−1∥2
F ≤16∥Y −Y ⋆∥2
F
σ2
min(Y ⋆)
for
i = 2, . . . , N −2. We now bound ∥W1 −R⊤
1 W ⋆
1 ∥2
F by


3
∥W1 −R⊤
1 W ⋆
1 ∥2
F
= ∥W ⋆
N · · · W ⋆
2 R1W1 −W ⋆
N · · · W ⋆
1 ∥2
F
≤2∥W1∥2∥WN · · · W2 −W ⋆
N · · · W ⋆
2 R1∥2
F
+2∥WN · · · W1 −W ⋆
N · · · W ⋆
1 ∥2
F
≤20∥Y ⋆∥2∥Y −Y ⋆∥2
F
σ2
min(Y ⋆)
,
(10)
where the second inequality uses the fact that Y
=
WN · · · W1X and ∥AX∥F = ∥A∥F for any A since X is
whitened. Based on the preceding discussion and the definition
of dist2(W, W⋆), we can conclude (7).
Finally, we can prove the other direction by
∥Y −Y ⋆∥2
F
= ∥
N
X
i=1
W ⋆
N · · · W ⋆
i+1Ri(Wi−R⊤
i W ⋆
i Ri−1)Wi−1 · · · X∥2
F
≤N
 N
X
i=2
9∥Y ⋆∥2
4
∥Wi −R⊤
i W ⋆
i Ri−1∥2
F
+∥W1 −R⊤
1 W ⋆
1 ∥2
F

≤9N
4 dist2(W, W⋆).
(11)
Main results To establish the convergence rate of RGD, we
require the loss function L to satisfy a certain property. Given
that our primary focus is the analysis of the local convergence,
we will assume that the loss function behaves well only in
a local region. Specifically, we will consider a category of
loss functions that satisfies the so-called restricted correlated
gradient (RCG) condition [29]:
Definition 2: We say the loss function L(·; Y ⋆) satisfies
RCG(α, β, C) condition for α, β > 0 and the set C if
⟨∇L(Y1; Y ⋆) −∇L(Y2; Y ⋆), Y1 −Y2⟩
≥α∥Y1 −Y2∥2
F + β∥∇L(Y1; Y ⋆) −∇L(Y2; Y ⋆)∥2
F (12)
for any Y1, Y2 ∈C.
The RCG condition is a generalization of the strong convexity.
When L represents the MSE loss, i.e., L(Y , Y ⋆) = ∥Y −
Y ⋆∥2
F , which is commonly used in the convergence analysis
of training deep linear networks [22, 23, 31], it satisfies the
RCG condition with α = β = 1 and C = Rdy×n. The RCG
condition may also accommodate other loss functions such as
the cross entropy (CE) loss.
Based on Definition 2, we can initially deduce the Rieman-
nian regularity condition as an extension of the regularity
condition found in matrix factorization [32, 33], ensuring
that gradients remain well-behaved within a defined region.
Specifically, we have
Lemma 2: (Riemannian regularity condition) Suppose the
training data (X, Y ⋆) is generated according to the data model
in Definition 1. Also assume that the loss function L in (2)
adheres to the RCG(α, β, C) condition where C ≜{Y : ∥Y −
Y ⋆∥2
F ≤
αβσ2
min(Y ⋆)
72(2N−1)2(N2−1)κ2(Y ⋆)}. Under this assumption,
for any W ∈{W : dist2(W, W⋆) ≤
αβσ2
min(Y ⋆)
9(2N−1)(N 2−1)}, the
function g in (2) satisfies the Riemannian regularity condition
as following:
N
X
i=2

Wi −R⊤
i W ⋆
i Ri−1, PTWiSt(∇Wig)

+⟨W1 −R1W ⋆
1 , ∇W1g⟩
≥
α
16(2N −1)κ2(Y ⋆)dist2(W, W⋆) +
β
(9N −5)∥Y ⋆∥2
×
 N
X
i=2
∥PTWiSt(∇Wig)∥2
F + ∥Y ⋆∥2∥∇W1g∥2
F

. (13)
Proof
To begin with, we can derive ∥W1∥2 ≤2∥W ⋆
1 ∥2 +
2∥W1 −R⊤
1 W ⋆
1 ∥2 ≤2∥Y ⋆∥2 + 2dist2(W, W⋆) ≤2∥Y ⋆∥2 +
2αβσ2
min(Y ⋆)
9(2N−1)(N2−1) ≤9∥Y ⋆∥2
4
where αβ ≤1
4 [29] is used .
Next, through the gradients ∇Wig
=
W ⊤
i+1 · · · W ⊤
N
∇L(Y ; Y ⋆)X⊤· · · W ⊤
i−1, i ∈[N], we need to derive
∥∇W1g∥2
F ≤∥∇L(Y ; Y ⋆) −∇L(Y ⋆; Y ⋆)∥2
F , (14)
∥∇Wig∥2
F ≤∥W1∥2∥∇L(Y ; Y ⋆)∥2
F
≤9∥Y ⋆∥2
4
∥∇L(Y ; Y ⋆) −∇L(Y ⋆; Y ⋆)∥2
F ,
(15)
where we employ ∇L(Y ⋆; Y ⋆) = 0. Combing (14) and (15),
we can obtain
N
X
i=2
∥PTWiSt(∇Wig)∥2
F + ∥Y ⋆∥2∥∇W1g∥2
F
≤
N
X
i=2
∥∇Wig∥2
F + ∥Y ⋆∥2∥∇W1g∥2
F
≤9N −5
4
∥Y ⋆∥2∥∇L(Y ; Y ⋆) −∇L(Y ⋆; Y ⋆)∥2
F , (16)
in which the first inequality follows from the fact that for any
matrix B = PTL(Xi)St(B)+P⊥
TL(Xi)St(B) where PTL(Xi)St(B)
and P⊥
TL(Xi)St(B) are orthogonal, we have ∥PTL(Xi)St(B)∥2
F ≤
∥B∥2
F .
Before analyzing the lower bound of cross term in (12), we
need to establish the upper bound for the inner product between
the orthogonal complement and the gradient as following:
T =
N
X
i=2
⟨P⊥
TWiSt(Wi −R⊤
i W ⋆
i Ri−1), ∇Wig⟩
≤
N
X
i=2
1
2∥Wi∥∥Wi −R⊤
i W ⋆
i Ri−1∥2
F
×∥∇L(Y ; Y ⋆) −∇L(Y ⋆; Y ⋆)∥F ∥W1∥
≤β
4 ∥∇L(Y ; Y ⋆) −∇L(Y ⋆; Y ⋆)∥2
F
+9(N −1)∥Y ⋆∥2
16β
N
X
i=2
∥Wi −R⊤
i W ⋆
i Ri−1∥4
F
≤β
4 ∥∇L(Y ; Y ⋆) −∇L(Y ⋆; Y ⋆)∥2
F
+ 9(N −1)
16β∥Y ⋆∥2 dist4(W, W⋆),
(17)
where P⊥
TWiSt(Wi −R⊤
i W ⋆
i Ri−1) =
1
2Wi(Wi −R⊤
i W ⋆
i
Ri−1)⊤(Wi −R⊤
i W ⋆
i Ri−1) [30] and ∇L(Y ⋆; Y ⋆) = 0.
Let
us
now
introduce
the
notation
H
=
Y ⋆−
WN · · · W2R⊤
1 W ⋆
1 X + PN
i=1 WN · · · Wi+1(Wi −R⊤
i W ⋆
i


4
Ri−1)Wi−1 · · · W1X, enabling us to simplify the expression
of the cross term within (13). Then (13) can be rewritten as
N
X
i=2

Wi −R⊤
i W ⋆
i Ri−1, PTWiSt(∇Wig)

+⟨W1 −R⊤
1 W ⋆
1 , ∇W1g⟩
= ⟨∇L(Y ; Y ⋆) −∇L(Y ⋆; Y ⋆), Y −Y ⋆+ H⟩−T
≥α∥Y −Y ⋆∥2
F + β
4 ∥∇L(Y ; Y ⋆) −∇L(Y ⋆; Y ⋆)∥2
F
−9(N 2 −1)
16∥Y ⋆∥2 dist4(W, W⋆)
≥
α
16(2N −1)κ2(Y ⋆)dist2(W, W⋆) +
β
(9N −5)∥Y ⋆∥2
×
 N
X
i=2
∥PTWiSt(∇Wig)∥2
F + ∥Y ⋆∥2∥∇W1g∥2
F

, (18)
where the first inequality follows the RCG condition, (17) and
∥H∥2
F ≤9N(N−1)
8∥Y ⋆∥2 dist4(W, W⋆) which is established through
a mathematical transformation and the use of norm inequalities.
The detailed proof for the upper bound of ∥H∥2
F has been
omitted here due to space limitations. In the last line, we
leverage (16), Lemma 1 and dist2(W, W⋆) ≤
αβσ2
min(Y ⋆)
9(2N−1)(N2−1).
We note that according to (7) in Lemma 1, the set {W :
dist2(W, W⋆) ≤
αβσ2
min(Y ⋆)
9(2N−1)(N 2−1)} implies the region C. By
leveraging Riemannian regularity condition in Lemma 2 and
the nonexpansiveness property of the polar decomposition-
based retraction in [30, Lemma 1], we ultimately reach the
following conclusion:
Theorem 1:
In accordance with the identical conditions
outlined in Lemma 2, we assume the initialization satisfies
dist2(W(0), W⋆) ≤
αβσ2
min(Y ⋆)
9(2N−1)(N2−1). When employing the
learning rate µ ≤
2β
(9N−5)∥Y ⋆∥2 and γ = ∥Y ⋆∥2 in RGD
(5), we have
dist2(W(t+1),W⋆)≤

1−ασ2
min(Y ⋆)µ
8(2N −1)

dist2(W(t),W⋆).
Our results reveal that the RGD demonstrates a linear
convergence rate with polynomial decay concerning N. In
addition, by Lemma 1, we can easily obtain ∥Y (t) −Y ⋆∥2
F ≤
(1−ασ2
min(Y ⋆)µ
8(2N−1) )t αβσ2
min(Y ⋆)
8N 2−12N+4. It is worth noting that, through
analogous analysis, Theorem 1 can be extended to a broader
scenario wherein W ⋆
j
takes on an arbitrary matrix, and
matrices W ⋆
i exhibit row orthogonality for i < j and column
orthogonality for i > j. Moreover, we emphasize that our focus
is primarily on the local convergence property of the RGD, and
does not cover initialization methods extensively investigated
in prior research, such as those discussed in [2, 21, 34, 35].
The research most closely related to our work is the
convergence analysis of gradient descent in deep linear
neural models across multiple layers, as demonstrated in
[22], wherein the MSE loss function is taken into account.
It is established that ∥Y (t) −Y ⋆∥2
F ≤ϵ can be deduced
when t ≥Ω( N 3∥Y ⋆∥6
F
c2
log( 1
ϵ )), provided that c ≤σmin(Y ⋆).
Applying a similar derivation as presented in [22, Theorem
1], utilizing both Theorem 1 and Lemma 1, we can infer that
t ≥Ω(N 2κ2(Y ⋆) log( 1
ϵ )) in the RGD is sufficient to meet
the same requirement. This further highlights the convergence
advantage of the RGD.
IV. EXPERIMENTS
In this section, we conduct experiments to compare the per-
formance of the RGD with gradient descent (GD). Specifically,
we concentrate on the multi-class classification task using the
MNIST dataset, where the input feature dimension is 784,
while the output feature is 10, represented as yi ∈R10. In
this representation, each yi is designed such that it holds a
value of 1 solely at the position that aligns with its categorical
label, leaving the other positions assigned to 0. For our model
architecture, we can deploy an multi-layer perceptron (MLP)
with N layers where W1 ∈R100×784, {Wi}N−2
i=2 ∈R100×100,
WN−1 ∈R50×100, and WN ∈R10×50. Each layer in the MLP
is connected by a linear or rectified linear unit (Relu) activation
function. While our theoretical result is only established for
linear networks, we will also test the performance on a
nonlinear MLP without bias terms.
We apply the GD and RGD for the CE loss function
[36] in combination with the softmax function to train MLP
models. The weight matrices are initialized using orthogonal
initialization [2], except for W1 in the RGD, which follows
a uniform distribution within the range of (−
1
√
784,
1
√
784)
[34]. In addition, we perform a grid search to fine-tune the
hyperparameters (µ and γ). In Figure 1, it is evident that RGD
0
20
40
60
80
100
Number of Iterations (t)
10-2
10-1
100
||Y(t)−Y⋆||2
F
||Y⋆||2
F
Linear-GD(4, 0.1)
Linear-GD(10, 0.02)
Linear-RGD(4, 3, 0.05)
Linear-RGD(10, 1, 0.05)
ReLU-GD(4, 1)
ReLU-GD(10, 0.3)
ReLU-RGD(4, 0.1, 10)
ReLU-RGD(10, 0.7, 10)
Fig. 1: Convergence analysis for GD(N, µ) and RGD(N, µ,
γ) with different activation functions and N.
achieves a quicker convergence compared to GD. Notably,
with an increase in the number of layers, the convergence
rate decreases in alignment with our theoretical analysis.
Moreover, due to the impact of nonlinear activation functions,
algorithms that employ ReLU demonstrate a comparatively
slower convergence rate compared to those utilizing linear
activation functions. However, despite this, the error of the
RGD with Relu outperforms that of the RGD using a linear
activation function.
V. CONCLUSION
In this letter, we have provided a convergence analysis
of the Riemannian gradient descent for a specific class of
loss functions within orthonormal deep linear neural networks.
Remarkably, our analysis guarantees a linear convergence rate,
provided appropriate initialization. This will serve as a stepping
stone for future explorations of training nonlinear orthonormal
deep neural networks with adaptive learning rates.


5
REFERENCES
[1] D. Mishkin and J. Matas, “All you need is a good init,” arXiv
preprint arXiv:1511.06422, 2015.
[2] A. M. Saxe, J. L. McClelland, and S. Ganguli, “Exact solutions
to the nonlinear dynamics of learning in deep linear neural
networks,” arXiv preprint arXiv:1312.6120, 2013.
[3] Y. Bengio, P. Simard, and P. Frasconi, “Learning long-term de-
pendencies with gradient descent is difficult,” IEEE transactions
on neural networks, vol. 5, no. 2, pp. 157–166, 1994.
[4] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of
training recurrent neural networks,” in International conference
on machine learning, pp. 1310–1318, Pmlr, 2013.
[5] Q. V. Le, N. Jaitly, and G. E. Hinton, “A simple way to initialize
recurrent networks of rectified linear units,” arXiv preprint
arXiv:1504.00941, 2015.
[6] M. Arjovsky, A. Shah, and Y. Bengio, “Unitary evolution
recurrent neural networks,” in International conference on
machine learning, pp. 1120–1128, PMLR, 2016.
[7] B. Hanin, “Which neural net architectures give rise to exploding
and vanishing gradients?,” Advances in neural information
processing systems, vol. 31, 2018.
[8] M. Harandi and B. Fernando, “Generalized backpropagation
etude de cas: Orthogonality,” arXiv preprint arXiv:1611.05927,
2016.
[9] S. Li, K. Jia, Y. Wen, T. Liu, and D. Tao, “Orthogonal deep
neural networks,” IEEE transactions on pattern analysis and
machine intelligence, vol. 43, no. 4, pp. 1352–1368, 2019.
[10] L. Huang, L. Liu, F. Zhu, D. Wan, Z. Yuan, B. Li, and L. Shao,
“Controllable orthogonalization in training dnns,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 6429–6438, 2020.
[11] J. Li, F. Li, and S. Todorovic, “Efficient riemannian optimization
on the stiefel manifold via the cayley transform,” in International
Conference on Learning Representations, 2020.
[12] J. Wang, Y. Chen, R. Chakraborty, and S. X. Yu, “Orthog-
onal convolutional neural networks,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recogni-
tion, pp. 11505–11515, 2020.
[13] F. Malgouyres and F. Mamalet, “Existence, stability and scala-
bility of orthogonal convolutional neural networks,” Journal of
Machine Learning Research, vol. 23, pp. 1–56, 2022.
[14] V. Dorobantu, P. A. Stromhaug, and J. Renteria, “Dizzyrnn:
Reparameterizing recurrent neural networks for norm-preserving
backpropagation,” arXiv preprint arXiv:1612.04035, 2016.
[15] Z. Mhammedi, A. Hellicar, A. Rahman, and J. Bailey, “Efficient
orthogonal parametrisation of recurrent neural networks using
householder reflections,” in International Conference on Machine
Learning, pp. 2401–2409, PMLR, 2017.
[16] E. Vorontsov, C. Trabelsi, S. Kadoury, and C. Pal, “On orthog-
onality and learning recurrent networks with long term depen-
dencies,” in International Conference on Machine Learning,
pp. 3570–3578, PMLR, 2017.
[17] M. Cisse, P. Bojanowski, E. Grave, Y. Dauphin, and N. Usunier,
“Parseval networks: Improving robustness to adversarial exam-
ples,” in International Conference on Machine Learning, pp. 854–
863, PMLR, 2017.
[18] M. Cogswell, F. Ahmed, R. Girshick, L. Zitnick, and D. Ba-
tra, “Reducing overfitting in deep networks by decorrelating
representations,” arXiv preprint arXiv:1511.06068, 2015.
[19] N. Bansal, X. Chen, and Z. Wang, “Can we gain more
from orthogonality regularizations in training deep networks?,”
Advances in Neural Information Processing Systems, vol. 31,
2018.
[20] L. Huang, X. Liu, B. Lang, A. Yu, Y. Wang, and B. Li,
“Orthogonal weight normalization: Solution to optimization over
multiple dependent stiefel manifolds in deep neural networks,”
in Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 32, 2018.
[21] P. Bartlett, D. Helmbold, and P. Long, “Gradient descent with
identity initialization efficiently learns positive definite linear
transformations by deep residual networks,” in International
conference on machine learning, pp. 521–530, PMLR, 2018.
[22] S. Arora, N. Cohen, N. Golowich, and W. Hu, “A convergence
analysis of gradient descent for deep linear neural networks,”
arXiv preprint arXiv:1810.02281, 2018.
[23] D. Zou, P. M. Long, and Q. Gu, “On the global convergence of
training deep linear resnets,” arXiv preprint arXiv:2003.01094,
2020.
[24] O. Shamir, “Exponential convergence time of gradient descent
for one-dimensional deep linear neural networks,” in Conference
on Learning Theory, pp. 2691–2713, PMLR, 2019.
[25] Z. Allen-Zhu, Y. Li, and Z. Song, “On the convergence rate
of training recurrent neural networks,” Advances in neural
information processing systems, vol. 32, 2019.
[26] S. Chatterjee, “Convergence of gradient descent for deep neural
networks,” arXiv preprint arXiv:2203.16462, 2022.
[27] M. Zhou, R. Ge, and C. Jin, “A local convergence theory
for mildly over-parameterized two-layer neural network,” in
Conference on Learning Theory, pp. 4577–4632, PMLR, 2021.
[28] X. Zhang, Y. Yu, L. Wang, and Q. Gu, “Learning one-
hidden-layer relu networks via gradient descent,” in The 22nd
international conference on artificial intelligence and statistics,
pp. 1524–1534, PMLR, 2019.
[29] R. Han, R. Willett, and A. R. Zhang, “An optimal statistical
and computational framework for generalized tensor estimation,”
arXiv preprint arXiv:2002.11255, 2020.
[30] X. Li, S. Chen, Z. Deng, Q. Qu, Z. Zhu, and A. Man-
Cho So, “Weakly convex optimization over stiefel manifold
using riemannian subgradient-type methods,” SIAM Journal on
Optimization, vol. 31, no. 3, pp. 1605–1634, 2021.
[31] Z. Zhu, D. Soudry, Y. C. Eldar, and M. B. Wakin, “The
global optimization geometry of shallow linear neural networks,”
Journal of Mathematical Imaging and Vision, vol. 62, pp. 279–
292, 2020.
[32] S. Tu, R. Boczar, M. Simchowitz, M. Soltanolkotabi, and
B. Recht, “Low-rank solutions of linear matrix equations
via procrustes flow,” in International Conference on Machine
Learning, pp. 964–973, PMLR, 2016.
[33] Z. Zhu, Q. Li, G. Tang, and M. B. Wakin, “The global
optimization geometry of low-rank matrix optimization,” IEEE
Transactions on Information Theory, vol. 67, no. 2, pp. 1308–
1331, 2021.
[34] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification,”
in Proceedings of the IEEE international conference on computer
vision, pp. 1026–1034, 2015.
[35] L. Xiao, Y. Bahri, J. Sohl-Dickstein, S. Schoenholz, and
J. Pennington, “Dynamical isometry and a mean field theory
of cnns: How to train 10,000-layer vanilla convolutional neural
networks,” in International Conference on Machine Learning,
pp. 5393–5402, PMLR, 2018.
[36] J. Zhou, C. You, X. Li, K. Liu, S. Liu, Q. Qu, and Z. Zhu,
“Are all losses created equal: A neural collapse perspective,” in
Advances in Neural Information Processing Systems, 2022.
