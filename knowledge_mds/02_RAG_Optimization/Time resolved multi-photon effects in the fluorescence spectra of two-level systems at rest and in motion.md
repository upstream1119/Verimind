# Time resolved multi-photon effects in the fluorescence spectra of two-level systems at rest and in motion

> Auto-converted plain text. Category: 02_RAG_Optimization

Time resolved multi-photon eﬀects in the ﬂuorescence spectra
of two-level systems at rest and in motion.
Emil Vi˜nas Bostr¨om,1 Andrea D’Andrea,2 Michele Cini,3 and Claudio Verdozzi1
1Division of Mathematical Physics and ETSF, Lund University, PO Box 118, 221 00 Lund, Sweden
2Istituto di Struttura della Materia, CNR, Area della Ricerca di Roma 1 (Montelibretti) Italy
3Dipartimento di Fisica, Universita‘ di Roma Tor Vergata,
Via della Ricerca Scientiﬁca 1, I-00133 Rome, Italy
(Dated: November 24, 2021)
We study the time-resolved ﬂuorescence spectrum in two-level systems interacting with an incident
coherent ﬁeld, both in the weak and intermediate coupling regimes. For a single two-level system in
the intermediate coupling case, as time ﬂows, the spectrum develops distinct features, that are not
captured by a semi-classical treatment of the incident ﬁeld. Speciﬁcally, for a ﬁeld on resonance with
the atomic transition energy, the usual Mollow spectrum is replaced by a four peak structure, and
for a frequency that is half of the atomic transition energy, the time-dependent spectrum develops a
second harmonic peak with a superimposed Mollow triplet. In the long-time limit, our description
recovers results previously found in the literature.
After analyzing why a diﬀerent behavior is
observed in the quantum and classical dynamics, the reason for the occurrence of a second harmonic
signal in a two-level system is explained via a symmetry analysis of the total (electron and photon)
system, and in terms of a three level system operating in limiting regimes. We ﬁnd an increased
second harmonic signal in an array of two-level systems, suggesting a superradiance-like enhancement
for multiple two-level systems in cavity setups. Finally, initial explorative results are presented for
two-level model atoms entering and exiting a cavity, which hint at an interesting interplay between
cavity-photon screening and atomic dynamics eﬀects.
I.
INTRODUCTION
Fluorescence, a type of luminescence [1–3], is a hall-
mark of quantum mechanics at work: A system that has
absorbed electromagnetic radiation re-emits it at a later
time, while the spins of the electrons involved in the de-
excitation process conform to speciﬁc selection rules.
In addition to being an operational mechanism in sev-
eral biological systems [4], ﬂuorescence serves in many
technologies of diﬀerent complexity, ranging from simple
indoor lighting to in-depth spectroscopic characterization
at the atomic scale. Investigations of ﬂuorescence started
before the advent of quantum mechanics, but it would be
the latter that provided the conceptual framework for a
microscopic description [1–3].
Spectroscopic methods have a rich history as a means
of investigating the internal structure of matter. In par-
ticular, with the development of ever more sophisticated
laser systems, the electronic dynamics in atomic and solid
state systems can now be mapped out in real time while
maintaining a high frequency resolution [5, 6]. This al-
lows to study in a precise manner the basic processes of
light-matter interaction, and even to characterize proper-
ties of light itself, as for example emitted via ﬂuorescence.
A minimal-complexity model to study ﬂuorescence
emission and ﬂuorescence spectroscopy is a two-level sys-
tem [7–9] (for example, a spinless electron that can be in
either of two nondegenerate quantum states, or in any
linear superposition thereof) interacting with a single ra-
diation mode via dipolar coupling. If the two levels are
thought to be selected from an atom, the model is also
referred to as a “two-level atom” (with the additional op-
tion that the center of mass of the atom can be either at
rest or in motion).
Two-level systems came into prominence with Rabi et
al.’s work for a magnetic moment exposed to a classical
circularly polarized ﬁeld [10]. In a subsequent study by
Bloch and Siegert the linearly polarized case was then
considered [11] (in this situation, the solution is more
complicated compared to Rabi’s original case [12]). The
next important development took place when the radia-
tion mode was also treated quantum-mechanically, and
the so-called rotating wave approximation (RWA) was in-
troduced [13–17]. Designed for weak-coupling and near-
resonance regimes, the RWA permits an explicit treat-
ment of the time dependence [18], and provides a conve-
nient route to the so-called dressed-level (or -atom) ap-
proach [9, 19, 20], where the levels of the system split and
are renormalized (shifted) by the radiation ﬁeld. In turn,
within this approach a clear picture emerges [21] of the
Mollow spectrum [22], a three-peak structure due to the
ﬂuorescent response of a two-level system to a resonant
or quasi resonant radiation mode.
Nevertheless, a number of interesting physical situa-
tions are outside the reach of RWA, as e.g. the intense
pulsed regime, where ﬁeld monochromaticity is absent
and oﬀ-resonant coupling cannot be avoided. The need
under some circumstances to go beyond the RWA has
in fact been recognized in several contexts (also by com-
paring exact and RWA solutions [23, 24]). For example,
when discussing modiﬁcations of the shape of the three
peaks in Mollow spectra [25–27], spontaneous emission in
three-level systems [28], or when center of mass dynamics
is included [29] (for a recent review, see e.g. [30]).
As these few, incomplete remarks suggest, two-level
systems coupled to radiation in diﬀerent “ﬂavors” remain
arXiv:2006.11673v1  [quant-ph]  20 Jun 2020


2
of capital relevance to this day to probe and redeﬁne the
knowledge boundaries in (quantum) optics [31–33]. This
can occur via generalization of the basic model(s) to-
gether with deeper mathematical analysis (see e.g. [34–
36]), to address unexplored coupling regimes [37, 38], or
novel areas of applications. For example, cavity quan-
tum optics and the Unruh eﬀect [29], quantum mechan-
ical interference [39], two-photon relaxation [40], quan-
tum phase transition [41], interaction of a photons matter
qubits [42] and Mollow spectra in ultracold atoms [43].
The quantum nature of light manifests in a clear way at
low photon number and for large light-matter coupling.
These two aspects contribute distinctly. This is diﬀer-
ent from the strong-ﬁeld regime, where a semiclassical
treatment becomes appropriate and where the eﬀective
coupling parameter is the product of ﬁeld strength and
coupling strength [12]. Concerning the few-photon limit,
this can be e.g. reached in high quality-factor cavities
[44–46]. On the other hand, to attain the strong cou-
pling (also denoted polaritonic) regime, a possibility is
oﬀered by the insertion of a quantum well into a dis-
tributed Bragg reﬂector cavity [47], i.e. by coupling the
photon ﬁeld and an optical inter-band transition (a Wan-
nier exciton).
Scope of this work.- In the present study we consider
two-level, one-electron systems interacting with two opti-
cal modes (the coherent-pump and a de-excitation ﬁeld).
Speciﬁcally, we address the (so far largely unexplored)
multi-photon eﬀects in ﬂuorescence spectra, that depend
separately on ﬁeld intensity and light-matter coupling
strength. This will be done in situations of progressive
complexity:
a single two-level atom at rest, an array
of two-level atoms at rest, and ﬁnally a single two-level
atom moving through an optical cavity. To this end, it
is necessary to employ a theoretical framework suitable
for both non-linear eﬀects (in relation to certain experi-
mental setups [48–50]) and an explicitly time-dependent
light-matter coupling.
Several years ago, three of the present authors intro-
duced an exact solution method [51, 52] for a large class
of multi-photon spectroscopy models. Their method is
based on a recursion technique (see e.g. Ref. [53]) in the
frequency domain, i.e. for the stationary limit of ﬂuores-
cence. They also pointed out the diﬀerence between the
exact solutions and the RWA. The aim was to address
systems where interactions with the environment are as
weak as possible. That is, where both inhomogeneous
(such as due to static and dynamical disorder, mode leak-
age, etc. ) and homogeneous (atomic collisions, but also
non-radiative decay, etc.) decoherence factors plays a mi-
nor role. Put diﬀerently, the focus was on a regime where
both energy-dependent broadening (not considered in the
rest of this paper, because assumed to be controllable),
and energy-independent one (denoted by Γ and retained
in the paper) are as small as possible. As discussed above,
current experimental capabilities provide practical and
close-to-ideal realisations of these premises with optical
cavity setups, and make time-resolved studies possible.
N
|2⟩
|1⟩
ϵ
ωa, ga
ωb, gb
(a)
· · ·
(b)
L
p
(c)
FIG. 1. Schematic of the systems considered in the paper.
In panel (a), a single two-level system of transition energy ϵ
interacting with a coherent ﬁeld of frequency ωa (with cou-
pling strength ga) and a ﬂuorescent ﬁeld of frequency ωb (with
coupling strength gb). In panel (b), an array of N two-level
systems interacting with a coherent ﬁeld and ﬂuorescent ﬁeld.
In panel (c), a two-level atom of momentum p passing through
a cavity of length L where it interacts with a coherent cavity
ﬁeld and emits ﬂuorescent photons.
In this way, it is possible to investigate the actual devel-
opment of the ﬂuorescence signal before the steady state
signal sets in.
With these considerations in mind, here we take a dif-
ferent methodological route from that in Refs. [51, 52], by
working in a real-time (and again free of RWA) picture,
and computing the exact time-dependent ﬂuorescence re-
sponse. To establish the eﬀect of multi-photon contri-
butions and counter-rotating terms, we specialize to the
Mollow regime (on-resonance situation) and to second-
harmonic generation (SHG) (oﬀ-resonance situation). In
the literature, Mollow spectra are discussed in terms of
two level systems [22], and thus our work conforms to
previous treatments. In contrast, SHG is commonly dis-
cussed in terms of three level systems [54]; however, we
will show that a genuine two-level system admits SHG.
Overall, our study here can thus be summarized as an ex-
ploration of multi-photon eﬀects in Mollow and SHG ﬂu-
orescence spectra across three diﬀerent two-level-system
setups (a single system at rest, many systems at rest, a
single system in motion).
A remark about the units used in this work: unless
otherwise stated at speciﬁc points in the paper, the en-
ergy unit is ϵ = ϵ2 −ϵ1, the distance between the two
levels in the system, and the time unit is ℏ/ϵ [55].
Organisation of the paper.- The rest of this article is
organized in three parts: In Sec. II we study the time-
dependent ﬂuorescence spectrum of an isolated two-level
system (see Fig. 1a). In Sec. III, we consider an array
of two-level systems interacting with a common coherent
ﬁeld (see Fig. 1b). In this case, the ﬂuorescence signal
shows an enhancement compared to the single two-level
system case, consistently with a superradiance-like mech-
anism. Finally, in Sec. IV we consider a two-level model
atom passing through an optical cavity (see Fig. 1c). We
explicitly treat the quantum motion of the atomic cen-


3
ter of mass, and show that this results in a ﬂuorescence
spectrum which diﬀers from that obtained from a semi-
classical treatment of the atomic motion. Conclusive re-
marks and outlook are in Sec. V.
II.
A SINGLE TWO-LEVEL SYSTEM
We start with the simplest of the three situations,
i.e.
a single two-level system interacting with two op-
tical modes. After presenting model and method of solu-
tion, we investigate the time evolution and the long time
limit of the system’s ﬂuorescence spectrum in the Mollow
regime. We consider the case when the Rabi frequency g
becomes a moderate fraction of the level spacing ϵ of the
two-level system (about 10 %). The results show that
for a ﬁeld in resonance with the atomic transition en-
ergy, the usual three peak Mollow spectrum is replaced
by a four peak structure. Instead, when the frequency is
half of the atomic transition energy, we obtain an SHG
spectrum with a superimposed Mollow splitting.
The
emergence of an SHG signal in a two-level system is an-
ticipated by the analysis of the spectrum of a three-level
system, and further validated by a symmetry analysis of
the coupled electron-photon states.
A.
Model and method
Our two-level system interacting with an incident and
a ﬂuorescent light-ﬁeld [51, 52] is described by the fol-
lowing Hamiltonian:
ˆH(t) = ˆHa + ˆHr + ˆHi(t).
(1)
We assume that the atom is occupied by a single spinless
electron, so that
ˆHa = ϵ1ˆc†
1ˆc1 + ϵ2ˆc†
2ˆc2 = ϵˆσz.
(2)
Here ˆci destroys an electron in the orbital |i⟩with energy
ϵi, ˆσz is the z-component Pauli operator, and ϵ = ϵ2 −ϵ1.
The free radiation modes are described by the Hamilto-
nian
ˆHr = ωaˆa†ˆa + ωbˆb†ˆb,
(3)
where ˆa annihilates a photon of the incident ﬁeld with
frequency ωa, which we assume to be in a coherent state
deﬁned by ˆa|α⟩= α|α⟩. Similarly ˆb annihilates a pho-
ton of the ﬂuorescent ﬁeld with frequency ωb. The light-
matter interaction Hamiltonian is given by
ˆHi(t) =
h
ga(t)(ˆa† + ˆa) + gb(t)(ˆb† + ˆb)
i
(ˆc†
1ˆc2 + ˆc†
2ˆc1)
=
h
ga(t)(ˆa† + ˆa) + gb(t)(ˆb† + ˆb)
i
ˆσx,
(4)
where ga(t) and gb(t) are the (time-dependent) couplings
of the electron to the incident and ﬂuorescent ﬁelds re-
spectively, and can have any time-dependence. Also, ˆσx
is the x-component Pauli operator. In the following we
consider the case gb(t) = gbe−Γt, which introduces a fre-
quency independent, phenomenological damping of rate
Γ. The use of Γ takes into account in a qualitative way ef-
fects left out, e.g. non-radiative transitions and/or mode
leakages in a cavity geometry (see also Appendix A 1).
Concerning the role of spontaneous decay in our de-
scription, we note that in the stationary regime (e.g. due
to a steady photon pump) it is often legitimate to over-
look this type of decay with respect to the stimulated
one.
Away from stationarity, other factors come into
play, depending on the situation: i) in the single-atom
case (and with Einstein’s description of radiation-matter
interaction as conceptual reference), spontaneous decay
does not induce a thermal bath; here, the coherence of
the overall optical response is not altered by neglecting
such decay (see e.g. [56]) ii) in the the many-atom case
(e.g the Dicke’s regime as discussed in Sect. III), thermal-
bath eﬀects and the signiﬁcance of spontaneous decay are
hindered by the collective eﬀect of super-radiance [57].
To describe the dynamics according to Eq. (1), we use
the exact conﬁguration interaction method. In this way,
the full wavefunction of the coupled atom-light system is
represented in the basis |i, n, m⟩≡|i⟩|n⟩|m⟩, with |i⟩the
state of the atom (where |1⟩is the ground state and |2⟩
the excited state), |n⟩a number state of the incident ﬁeld
and |m⟩a number state of the ﬂuorescent ﬁeld. We start
from the initial state |ψ0⟩= |1, α, 0⟩(here, the number
state |n⟩has been replaced by the coherent state |α⟩of
the pump ﬁeld), and time-evolve it with the full Hamilto-
nian H using the short iterated Lanczos technique [58, 59]
(see also Appendix A 2).
We note here that the recursion method originally
employed to study ﬂuorescence in the stationary limit
[51, 52] can be used for more general setups, e.g. to gen-
erate exact solutions with several electronic levels and
several bosonic modes [53]. However, its applicability is
not immediate for genuinely time-dependent Hamiltoni-
ans, and to address the transient response of a system.
Hence, the need to proceed here with a real-time ap-
proach.
It has been shown [60] that to make contact with exper-
imental time-resolved light spectra, the transition proba-
bility needs to be convolved with the resolution function
of a Fabry-Perot spectrometer. Also, depending on the
experiment performed on a quantum system, a detection
of N-photon correlations can be used [61].
However, here we employ a deﬁnition of the spectrum
diﬀerent from the one considered in [60, 61]. Namely, we
consider the probability P that at least one photon (at a
given frequency) is emitted. Such a spectrum would not
be obtained by an interferometer, Fabry-Perot or other-
wise, but rather by an apparatus including e.g. a prism
and a photomultiplier.
In other words, this we would
correspond to measure particle-like photons, rather than
waves, and probing simultaneously the atomic state.
Accordingly, our observable of main interest is the
probability to ﬁnd m photons in the ﬂuorescent ﬁeld of


4
(a)
(b)
(c)
(d)
FIG. 2. Fluorescence spectra of a two-level system interacting
with a coherent ﬁeld with frequency ωa = ϵ and an average
number of photons α2 = 25 [panels (a) and (c)] and α2 = 1
[panels (b) and (d)]. The light-matter coupling ga is chosen so
that gaα = 0.1 in both cases, giving ga = 0.02 and ga = 0.1
respectively. The coupling to the ﬂuorescent ﬁeld is gb = 0.01.
Panels (a) and (b) show the asymptotic spectrum P(ω) =
P(t →∞, ω) obtained using either a quantized coherent ﬁeld
or taking the semi-classical limit of Eq. (6). Panels (c) and
(d) show the corresponding time-dependent spectrum using a
quantized coherent ﬁeld. The units of energy and time are
respectively given by ϵ = ϵ2 −ϵ1 and ℏ/ϵ.
frequency ωb at time t, given by
Pm(t, ωb) =
X
ni
|⟨i, n, m|T
h
e−i
R t
0 ˆ
H(t′)dt′i
|1, α, 0⟩|2, (5)
where the dependence on ωb in RHS is implicitly con-
tained in ˆH(t). Since gb(t) has an exponential decay, P
will be independent of t in the long time limit. In the
following we will focus on the quantity P = P
m>0 Pm,
giving the probability that at least one ﬂuorescent photon
has been emitted.
The semi-classical limit of our model is obtained by
taking α →∞and g →0, while keeping gα constant. In
this limit ˆHr = ωˆb†ˆb and the interaction Hamiltonian is
ˆHsc
i (t) =
h
2gaα cos(ωat) + gb(t)(ˆb† + ˆb)
i
(ˆc†
1ˆc2 + ˆc†
2ˆc1).
(6)
The t →∞limit of the model has been studied in
previous works [51, 52].
In that case, the Hamilto-
nian was split as ˆH(t) = ˆH′
0(t) + ˆH′(t) with ˆH′(t) =
gb(t)(ˆc†
1ˆc2+ˆc†
2ˆc1)(ˆb†+ˆb), and the probability P1 was eval-
uated under the assumption that ˆH′(t) acts only once
during the time evolution. Physically this corresponds
to a ﬁrst-order treatment of the ﬂuorescent ﬁeld, in con-
trast to the exact numerical solution of the present work,
which retains the eﬀects of the interaction at all orders.
B.
Two-level system and Mollow spectra
In the following we let the transition energy ϵ = 1 de-
ﬁne our unit of energy, and further ﬁx the parameters
gb = 0.01 and Γ = 0.02. We start by studying the ﬂuo-
rescence spectrum for ωa = 1 as a function of α and ga,
keeping the product gaα ﬁxed. The results are displayed
in Fig. 2, and show both the asymptotic spectrum (as
t →∞) and the explicit time-evolution. For α = 5 and
ga = 0.02 we see the well-known Mollow spectrum, that
is qualitatively reproduced by the semi-classical approxi-
mation. Keeping the product gaα ﬁxed and taking α = 1
and ga = 0.1 the semi-classical result is unchanged, while
the full quantum treatment gives a ﬂuorescence spectrum
with four peaks and additional substructure.
To understand the spectra we consider the states |i, n⟩
with the atom in state i and n photons in the incident
ﬁeld. For ωa = ϵ the levels |1, n⟩and |2, n −1⟩are degen-
erate, and mixed by the light-matter interaction. Diago-
nalizing the Hamiltonian in this subspace (i.e. neglecting
the counter-rotating terms), we ﬁnd the energies
ϵ±,n = ϵ1 + ϵ2
2
+ (n −1/2)ωa ± ga
√n.
(7)
For a coherent state with large α the Poisson distribution
is sharply peaked around α2, so that n ∼α2 and the
energies diﬀerences between successive n are
ϵ±,n+1 −ϵ∓,n = ωa ± ga
√
n + 1 ± ga
√n
n→|α|2
−−−−−→±2gaα
ϵ±,n+1 −ϵ±,n = ωa ± ga
√
n + 1 ∓ga
√n
n→|α|2
−−−−−→0. (8)
In this limit the energy splittings are independent of n,
and transitions between successive levels give a three-
peaked structure.
For small α, this is no longer the
case, since the splitting ϵ+,n −ϵ−,n = 2ga
√n. The non-
uniformity in the level spacing is a clear sign of the quan-
tum nature of the light (the low photon number limit),
and is e.g. responsible for photon blockade eﬀect [62, 63].
In the present context, it gives rise to the additional fea-
tures observed in the ﬂuorescence spectrum of Fig. 2.
C.
Prelude to SHG in a two level system:
frequency doubling in an ordinary three-level system
Before addressing SHG in a two-level system, we make
a detour into the more familiar theory of SHG in three-
level systems. Let the Hamiltonian be ˆH(t) = ˆHe + ˆHr +
ˆHi(t), with ˆHr as in previous sections, and the electronic
Hamiltonian be
ˆHe = ϵ1ˆc†
1ˆc1 + ϵ2ˆc†
2ˆc2 + ϵ3ˆc†
3ˆc3.
(9)
For the light-matter interaction we consider two scenar-
ios, illustrated in Fig. 3. In the ﬁrst the incident ﬁeld
couples to the transitions |1⟩↔|2⟩and |2⟩↔|3⟩, and


5
the ﬂuorescent ﬁeld to the transition |1⟩↔|3⟩.
The
interaction Hamiltonian is then
ˆH(1)
i
(t) = f(t)(ˆa† + ˆa)(ˆc†
1ˆc2 + ˆc†
2ˆc3 + H.c.)
+ gb(t)(ˆb† + ˆb)(ˆc†
1ˆc3 + ˆc†
3ˆc1).
(10)
In the second case the incident ﬁeld also couples to the
transition |1⟩↔|3⟩, with a strength g(t), and the Hamil-
tonian is
ˆH(2)
i
(t) = ˆH(1)
i
(t) + ga(t)(ˆa† + ˆa)(ˆc†
1ˆc3 + ˆc†
3ˆc1).
(11)
If the levels |1⟩and |3⟩are of deﬁnite and diﬀerent sym-
metry, while the level |2⟩is assumed to be of mixed sym-
metry, both these models allow SHG using a perturba-
tive treatment. However, in the case of a parity-invariant
electronic Hamiltonian, where all electronic states have
deﬁnite parity, both models forbid SHG in a perturba-
tive approximation. The reason is that there is no way
to arrange the parities such that both the excitation and
emission steps are allowed: the parities π1 and π3 of lev-
els |1⟩and |3⟩need to be diﬀerent (for the ﬂuorescent
transition to be allowed), while the parity π2 of level |2⟩
needs to be diﬀerent both from π1 and π3 (in order for
the exciting transitions to be allowed).
We now study the ﬂuorescence spectrum in the non-
perturbative limit. We take ϵ2 −ϵ1 = 0.5 and ϵ3 −ϵ1 = 1,
and let the incident ﬁeld be resonant with the transition
energies (ωa = 0.5). We choose the couplings as f(t) =
ga(t) = 0.1, take α = 1, and as before let gb = 0.01
and Γ = 0.02. The results are reported in Fig. 3. For the
Hamiltonian ˆH(1)
i
we see a broadened SHG peak centered
around ωb = 1, while for ˆH(2)
i
the spectrum has two
contributions corresponding to Rayleigh scattering and
SHG centered at ωb = 0.5 and ωb = 1 respectively. As
expected, both models predict a non-zero SHG signal.
A connection with a simpler two-level system can be
made by taking the limit ϵ2 →∞, illustrated here by
considering ϵ2 −ϵ1 = 1, 1.5 and 2 (see Fig. 3). With the
interaction ˆH(1)
i
the ﬂuorescence signal narrows around
ω = 1 as ϵ2 −ϵ1 is increased, since the coupling between
degenerate states causing the broadening decreases as the
levels are energetically separated (cf. the discussion of
the Mollow spectrum). For even larger values of ϵ2 −ϵ1
(not shown), the peak tends to zero since excitation of the
atom becomes increasingly unlikely. With the interaction
ˆH(2)
i
both the Rayleigh and SHG peaks narrow as ϵ2 −
ϵ1 is made large. However, due to the presence of the
coupling between the incident ﬁeld and the transition
|1⟩↔|3⟩, a ﬁnite ﬂuorescence signal remains even for
ϵ2 −ϵ1 →∞, i.e when eﬀectively the Hamiltonian ˆH(2)
i
collapses onto that of the two-level system. The speciﬁc
shape of the SHG proﬁle is discussed in the next Section.
As a ﬁnal consideration, we note that in quantum op-
tics it is sometimes useful to perform an adiabatic elimi-
nation of intermediate levels, to end up with a reduced-
space eﬀective Hamiltonian [64, 65]. In appendix B, such
a reduction is performed to map the three-level system
|1⟩
|3⟩
|2⟩
ωa
ωb
(a)
|1⟩
|3⟩
|2⟩
ωa
ωb
(b)
FIG. 3. Fluorescence spectra for a three-level system inter-
acting with a coherent ﬁeld with a frequency ωa = 1 and an
average number of photons α2 = 1. The coupling between
the system and the coherent ﬁeld is given by f = ga = 0.1,
while the coupling to the ﬂuorescent ﬁeld is gb = 0.01. The
energy levels of the atom are ϵ1 = 0 and ϵ3 = 1, with ϵ2 taking
the values 0.5, 1, 1.5 and 2. In panel (a) the coherent ﬁeld is
coupled to the transitions 1 ↔2 and 2 ↔3, while in panel
(b) the coherent ﬁeld in addition couples to transition 1 ↔3.
The couplings in the system are indicated schematically in
the panels to the left. For ease of reading, the z-axis has been
multiplied by 100. The unit of energy is given by ϵ = ϵ3 −ϵ1.
of Fig. 3 onto a two-level one [66], and to see if SHG can
be exactly described in a two-level system in the limit
ϵ2 →∞of a three level system. The results in the ap-
pendix show that this is not the case.
In fact, going
beyond the RWA and neglecting oﬀ-resonant (but dipole-
allowed) transitions in the system are two elements which
play distinct roles.
Speciﬁcally, the occurrence of the
Mollow structure is not only due to the removal of the
RWA, but also due to the explicit inclusion of the virtual
state in the total Hamiltonian.
Preventing transitions
to this state removes the Mollow structure, also without
RWA and for strong light-matter coupling.
D.
Two-levels again: Second harmonic generation
With the information gained so far, we ﬁnally move to
study the ﬂuorescence spectrum in a two level system as
described in Eqs. (1-4) [67]. Since a common theme of
this work is ﬂuorescence in the SHG regime, we consider
the case of ωa = ϵ/2, with all other parameters as in
Sections. II A, II B.
In Fig. 4 we show results for α = 1 and α = 5, while
keeping gaα = 0.1 ﬁxed. For a small light-matter cou-


6
(a)
(b)
(c)
(d)
FIG. 4. Fluorescence spectra of a two-level system interact-
ing with a coherent ﬁeld with a frequency ωa = ϵ/2 and an
average number of photons α2 = 25 [panels (a) and (c)] and
α2 = 1 [panels (b) and (d)]. The light-matter coupling ga is
chosen so that gaα = 0.1 in both cases, giving ga = 0.02 and
ga = 0.1 respectively. The coupling to the ﬂuorescent ﬁeld is
gb = 0.01. Panels (a) and (b) show the asymptotic spectrum
P(ω) = P(t →∞, ω) obtained using either a quantized coher-
ent ﬁeld or taking the semi-classical limit of Eq. (6). Panels
(c) and (d) show the corresponding time-dependent spectrum
using a quantized coherent ﬁeld. The units of energy and time
are respectively given by ϵ = ϵ2 −ϵ1 and ℏ/ϵ.
pling the semi-classical calculation is seen to be in good
agreement with the quantum treatment, while for large
coupling the results diﬀer by the presence of a superim-
posed Mollow spectrum on the second harmonic peak.
This additional feature can be understood similarly to
the Mollow spectrum of Fig. 2, as due to an energy level
splitting depending on the coupling strength ga and pho-
ton number α2. The energy levels are shown in Fig. 5,
and we see that for small ga and large α the splittings
are small and uniform (between successive levels), while
for large ga and small α the splittings are large and non-
uniform. We note that for small ga, in contrast to the
Mollow spectrum discussed above, the level splitting is
expected to be ∼g2
a to leading order, since the cou-
pling between the levels is of second order in the in-
teraction. To summarize, i) the ability of the quantum
treatment to discriminate between photon number and
coupling strength (they always appear via their product
in the semiclassical treatment) and ii) the photon ﬂuctu-
ations at low photon number, are likely the reasons for
the Mollow structure in the quantum SHG signal. This is
in line with the evidence provided by the three-level sys-
tem results in Fig. 3 where, depending on the closure of
the “three-level triangle” via the pump ﬁeld, the Mollow
structure in the SHG peak is observed or not.
For the results just presented the photon energy ℏωa of
the coherent ﬁeld is commensurate with the atomic tran-
sition energy, i.e. ℏωa = ϵ/n. Although our approach
is completely general and we can in principle study any
frequency, a systematic scan of the parameters is outside
the scope of the of the present work. However, we men-
tion that according to additional calculations (not shown)
the spectra for frequencies ℏωa/ϵ ≈0.3 −0.8 resemble
the second harmonic spectrum above, although with the
position of the Rayleigh peak displaced to ℏωb ≈ℏωa.
Instead, for frequencies ℏωa/ϵ ≈0.9 −1.1 the Rayleigh
and harmonic peaks blend together to form the four peak
Mollow structure discussed above.
Parity conservation.- Although borne by the exact
numerical results of Sect. II D (and also supported by
the connection to three-level physics as discussed in
Sect. II C), the occurrence of a SHG in a two-level system
might remain at some extent counter-intuitive. For sec-
ond harmonic generation (SHG) to occur, two photons
are needed to excite the atom. This requires atomic lev-
els of equal parity, since in the low intensity limit where
the absorption process is well described by perturbation
theory, the two-photon absorption probability otherwise
vanishes. However, the emission of a double frequency
photon requires the atomic levels to have opposite par-
ity. This apparent paradox however vanishes at stronger
light-matter coupling, where a more appropriate descrip-
tion of the system is in terms of dressed atomic levels.
In this regime the electronic states are mixed by the cou-
pling to the light ﬁeld, and no longer have deﬁnite parity.
To understand how SHG can happen in a two-level
system, we look at the parity of the eigenstates of ˆH. A
parity operator can be deﬁned through [68]
ˆΠ = (ˆn1 −ˆn2)eiπˆnaeiπˆnb,
(12)
where ˆna and ˆnb are the photon number operators of the
incident and ﬂuorescent ﬁelds respectively. It is straight-
forward to show that [ ˆH, ˆΠ] = 0, from which it follows
that the eigenstates of ˆH can be classiﬁed according to
the eigenvalues of ˆΠ. The general structure of the eigen-
states with the electron and the incident ﬁeld coupled
is [68]
|Ψk
e⟩=
X
n
ck
2n|1, 2n⟩+
X
n
ck
2n+1|2, 2n + 1⟩
(13)
|Ψk
o⟩=
X
n
ck
2n+1|1, 2n + 1⟩+
X
n
ck
2n|2, 2n⟩,
(14)
where e and o denote even- and odd-parity states re-
spectively. We see that an eigenstate with a well-deﬁned
parity in the coupled system has in general an undeﬁned
parity in the electronic and photonic subspaces. There-
fore an argument against SHG in the two-level system,
which relies on the conservation of only electronic parity
is not applicable.
Consider now the evolution of a system starting from
the initial state |Ψ⟩= |1, α⟩. Since the coherent state is a
superposition of number states, it explicitly breaks par-
ity symmetry. We therefore expect the time-evolution to
induce transitions between the initial state and all eigen-
states allowed by energy conservation, which in particu-
lar means that an electron may be excited to the upper


7
(a)
(b)
FIG. 5. Energy levels as a function of coupling strength for
a two-level system interacting with a single light ﬁeld.
In
both panels, the solid black line indicates the coupling ga and
the dashed line the average energy ϵav = ωaα2 of the setups
described in the main text. Panel (a) shows the energy levels
in a range around the coupling ga = 0.02 and energy ϵav for
ωa = 0.5 and α = 25, while panel (b) shows the energy levels
in a range around the coupling ga = 0.1 and energy ϵav for
ωa = 0.5 and α = 1. The unit of energy is given by ϵ = ϵ2−ϵ1.
atomic level. However, even for an initial state of deﬁnite
parity (as for example |Ψ⟩= |1, n⟩) the time-evolution
will mix eigenstates, but now in a deﬁnite parity sector.
Thus, in stark contrast with the predictions of per-
turbation theory, an SHG signal also occurs for parity
conserving dynamics. In addition, and diﬀerently from
a three-level system, the closure of the multi-photonic
triangle (which seems needed for the appearance of the
three-peaked Mollow structure), always occurs in a two
level system, as an emerging symmetry mixing behavior.
Furthermore, Mollow-like overtones can also be present
in higher-order harmonics [51, 52], a feature that could
become relevant in the time domain (e.g. in the ultrafast
regime). These summary notions provide rigorous foun-
dation and motivation to investigate SHG in more com-
plicated two-level system setups and out of the stationary
limit, i.e. when the eﬀects discussed so far will appear
within characteristic timescales (for example, when in-
vestigating an atom moving through a cavity of ﬁnite
length).
Accordingly, time-resolved multi-photon ﬂuo-
rescence is the main theme of the rest of the paper.
III.
AN ARRAY OF TWO-LEVEL SYSTEMS
We now consider N two-level systems interacting with
one incident and one ﬂuorescent light mode. This brings
in the possibility that the diﬀerent two-level systems in-
teract cooperatively with the radiation, with an enhance-
ment of the radiation ﬁeld that goes under the name of
superradiance [69] (more precisely, depending on the na-
ture of the initial state, one can speak of superradiance
or supercoherence [70]).
The standard Dicke model [69], which describes N two-
level systems interacting with a single optical mode, plays
an archetypal role in the study of superradiance and has
been the focus of extensive investigations (see e.g. [71–
74]).
An exact solution within the RWA of the Dicke
model at resonance has been known for quite some time
(for this reason, the model is also referred to as the Tavis-
Cummings model [75]) but some aspects related to this
model are still being debated [76]. A notable case con-
cerns the existence of a no-go theorem for superradiance,
also in connection to the role of dipolar couplings between
electrons and photons and the interactions among the
diﬀerent two-level systems. Furthermore, consideration
is needed for the superradiant behavior when N →∞,
i.e. the periodic array limit. Mathematically, the inten-
sity diverges in that situation, but in fact a continuous
electronic band structure emerges in the thermodynami-
cal limit, and the bulk polaritonic regime applies, with a
ﬁnite optical response [77].
These diﬀerent aspects are not addressed here. Rather,
our simple analysis is aimed to gain qualitative insight
into how the behavior of the single two-level system dis-
cussed above changes in a more complex setup. As be-
fore, we position ourselves in the fully-resonant and the
SHG regimes for the incident ﬁeld. We will consider the
problem from two diﬀerent but complementary perspec-
tives: (i) We ﬁrst study the equilibrium properties of
the system in the large-N limit using a generalization of
the Dicke model. (ii) Secondly, we investigate the non-
equilibrium properties of the system for ﬁnite N (similar
to what done earlier for N = 1) using an exact numerical
treatment. We ﬁnd that, compared to the single two-level
system case, the ﬂuorescence signal shows an enhance-
ment compatible with a superradiance-like mechanism,
both in the resonant and SHG regimes.
As an obvious extension of the N = 1 case considered
earlier, the total Hamiltonian becomes
H = ˆHa + ˆHf + ˆHi(t),
(15)
with
ˆHa =
2N
X
i=1
ϵiˆc†
i ˆci,
(16)
ˆHf = ωaˆa†ˆa + ωbˆb†ˆb
(17)
ˆHi(t) = ga(t)(ˆa† + ˆa)
N
X
i=1
(ˆc†
2i−1ˆc2i + ˆc†
2iˆc2i−1)
+ gb(t)(ˆb† + ˆb)
N
X
i=1
(ˆc†
2i−1ˆc2i + ˆc†
2iˆc2i−1).
(18)
Here ˆc2i−1 refers to the ground state and ˆc2i to the ex-
cited state of two-level system i. Due to the speciﬁc way
the ﬁelds couple to the two-level systems, we can rewrite


8
the Hamiltonian in a more compact form as
ˆH =
N
X
i=1
ωiˆσz,i + ω0ˆa†ˆa + ωˆb†ˆb
+
h
ga(t)(ˆa† + ˆa) + gb(t)(ˆb† + ˆb)
i N
X
i=1
ˆσx,i,
(19)
where ωi = ϵ2i −ϵ2i−1. When all excitation energies ωi =
ωs, the ﬁelds only couple to the total spin operators ˆSz =
P ˆσz,i and ˆSx = P ˆσx,i. The Hamiltonian then closely
resembles the Dicke Hamiltonian [69], and is written
ˆH = ωs ˆSz + ωaˆa†ˆa + ωbˆb†ˆb
+
h
ga(t)(ˆa† + ˆa) + gb(t)(ˆb† + ˆb)
i
ˆSx.
(20)
A.
Ground state
We start by discussing the ground state properties. For
an initial state with all two-level systems in their ground
state, we can further simplify the Hamiltonian using the
Holstein-Primakoﬀtransformation [78]. Writing the spin
operators in terms of bosonic creation and annihilation
operators, ˆSz = −(N/2)+ˆs†ˆs and ˆS+ = s†√
N −ˆs†ˆs, it is
straightforward to check that the commutation relations
deﬁning the spin algebra are preserved. Taking the limit
where N ≫⟨ˆs†ˆs⟩, the ˆS+ operator can be expanded as
ˆS+ ≈s†N + O(N −1), and the Hamiltonian becomes
ˆH = ωsˆs†ˆs + ωaˆa†ˆa + ωbˆb†ˆb
+ N
h
ga(t)(ˆa† + ˆa) + gb(t)(ˆb† + ˆb)
i
(ˆs† + ˆs),
(21)
and the system maps into three coupled oscillators.
For gb = 0, the Hamiltonian of Eq. (21) reduces to the
usual Dicke Hamiltonian, where a transition to a superra-
diant state takes place at a critical coupling ga ≈(2N)−1,
with the ﬁeld intensity of the radiation becoming pro-
portional to N 2 [76]. To see this, we follow closely and
reproduce here the discussion given in Ref. [76], starting
by writing the Hamiltonian in terms of canonical coordi-
nates:
ˆH = 1
2(ˆp2
s + ˆp2
a) + 1
2
 ˆxs ˆxa
 
ωs 2λ
2λ ωa
 
ˆxs
ˆxa

,
(22)
where, to keep the notation light, we have introduced
the coupling parameter λ = Nga.
We also assume
that the ﬁeld is resonant with the two-level systems, i.e.
ωa = ωs = 1. To identify the superradiant transition,
we look for the point λc where the number of photons
in the lowest normal mode of ˆH diverges. The lowest
mode is identiﬁed by moving to normal canonical coor-
dinates and momenta of ˆH, given by ˆx± = (ˆxs ± ˆxa)/
√
2
and ˆp± = (ˆps ± ˆpa)/
√
2, respectively, and with nor-
mal frequencies Ω2
± = 1 ± 2λ.
In the ground state,
the expectation value of ˆx2
± is ⟨ˆx2
±⟩= (2Ω±)−1, which
(a)
(b)
FIG. 6.
Panel (a):
the critical coupling strength for the
ground-state superradiant transition as a function of λa =
Nga and λb = Ngb. The green line refers to the resonant case
ωs = ωa = ωb = 1, while the blue line shows the non-resonant
case ωs = ωb = 1 and ωa = 1/2. Panel (b): expansion coeﬃ-
cients |ci|2 of the lowest eigenstate of Eq. (24) in the basis of
oscillators from Eq. (21). The dashed lines refer the resonant
case ωs = ωa = ωb = 1, while the solid lines show the non-
resonant case ωs = ωb = 1 and ωa = 1/2. The unit of energy
is given by ωs.
diverges at coupling λc, whilst ⟨p2
±⟩remains well be-
haved.
We then go back to the original coordinates
ˆxa/b, ˆpa/b, expressing them in terms of ˆx±, ˆp±, and note
that na = ⟨ˆa†ˆa⟩= (⟨ˆp2
a⟩+ ⟨ˆx2
a⟩−1)/2.
Then, when
λ →λc, the diverging contribution comes from ⟨ˆx2
±⟩, i.e.
na
≈
λ→λc
1
4
√
1 + 2λ +
1
4
√
1 −2λ
(23)
which diverges for |λ| = λc = 1/2. Thus, for λ > 0 we
can identify the ground state superradiant transition as
the point where the lowest eigenvalue Ω−vanishes.
To extend the above discussion from Ref. [76] to the
original Hamiltonian Eq. (21) with the ﬂuorescent ﬁeld
included, we write
ˆH = 1
2(ˆp2
s + ˆp2
a + ˆp2
b)
+ 1
2
 ˆxs ˆxa ˆxb



ωs
2λa 2λb
2λa
ωa
0
2λb
0
ωb




ˆxs
ˆxa
ˆxb

,
(24)
where again λa = Nga and λb = Ngb. At this point,
if we take ωs = ωa = ωb = 1, the lowest normal mode


9
eigenvalue Ω0 of H is given by
Ω0(λa, λb) = 1 −2
q
λ2a + λ2
b,
(25)
which is the generalization to the case of two ﬁelds both
in resonance with the N two-level systems. If instead, to
address the SHG regime, we assume that ωs = ωb = 1
and ωa = 1/2, we obtain
Ω0(λa, λb) = 1
12
"
11 −4β
 2
A
1/3
−4
A
2
1/3#
, (26)
where A = α −
p
α2 −4β3, α = 1/32 + 9λ2
a −18λ2
b,
and β = 1/16 + 12λ2
a + 12λ2
b. As before, we search for
signatures of a superradiant transition by looking at the
points where Ω0 vanishes. In the resonant case this is
easily done, and one obtains the semi-circle solution set
λa =
p
1/4 −λ2
b.
We note that for either λa = 0 or
λb = 0, we recover the critical coupling of the standard
Dicke model discussed above. In the non-resonant case,
we solve instead Eq. (26) numerically to ﬁnd the result
in Fig. 6a. Interestingly, we ﬁnd that the signature of a
superradiant transition occurs for smaller values of the
coupling when one of the ﬁelds is non-resonant with the
atomic transitions.
The above argument indicates that the lowest eigen-
state of the Hamiltonian undergoes a superradiant tran-
sition. However, we have not yet determined how this
state is related to the original photon ﬁelds, and there-
fore at this point it is not clear how these ﬁelds behave at
the transition. In analogy with the Dicke model, the aver-
age number of photons in each of the original light ﬁelds
(in the ground state) is proportional to |ca/b|2(2Ω0)−1,
where |ci|2 = |⟨i|Ω0⟩|2 are the projections of the normal
mode |Ω0⟩onto the original oscillators. The ﬁelds should
therefore undergo a transition to a superradiant state for
values of λa and λb such that (i) Ω0 = 0, and (ii) |ca/b|2
are ﬁnite. The coeﬃcients are found by numerical diago-
nalization of the Hamiltonian in Eq. (24), and shown in
Fig. 6b for values of λa and λb at the superradiant tran-
sition. It is apparent that for λa/b > 0 we always have
|ca/b|2 > 0, so that condition (ii) above is always satis-
ﬁed. We thus ﬁnd that at the transition points found
above, both the incident and the coherent ﬁelds behave
as if a superradiant state is attained.
B.
Real-time simulations for ﬁnite N
Having discussed some ground state features of a two-
level system array coupled to radiation, we now return
to explore the time evolution of the system deﬁned by
Eq. (19). We start from an initial state with all atoms
in their ground state, and therefore we do not expect to
see a superradiant emission burst. However, we are inter-
ested in exploring how the ﬂuorescence spectrum changes
as we approach coupling strengths close to the (equilib-
rium) superradiant transition. We consider the parame-
ters ωs = ωb = 1, ωa = 0.5, and choose |α|2 = M = 9 for
the average number of photons in the cavity. With M of
the same of order as the number of atoms N, the energy
of the ﬁeld should be enough to simultaneously excite all
the atoms. This means that for a two-level array with
bare couplings given by e.g. ga = 0.03 and gb = 0.01
(to be used below in the actual simulations), the min-
imal value of N needed to get the critical values of λa
and λb indicated in Fig. 6 is given by N ≈14. This is
done by checking for which value of N that λa = Nga
and λb = Ngb cross the blue line in Fig. 6a.
Since the arguments of the previous section are only
strictly valid in the ground state and for large N (due to
the use of the Holstein-Primakoﬀtransformation), there
is no guarantee that they would hold in real time. Thus,
our estimate for N just provides a hint of the order mag-
nitude of N where we can expect superradiant eﬀects
to appear. In addition, the non-equilibrium signatures of
superradiance are typically expressed through the scaling
of the duration and intensity of the superradiant burst
with the number of two-level systems N, given respec-
tively by 1/N and N 2. We therefore consider below the
ﬂuorescence spectrum for N ∈{1, 2, . . . , 10}, and look
for signatures consistent with these scaling laws.
In Fig. 7 we show the ﬂuorescence spectra for an array
of N = 2 and N = 10 two-level systems. We ﬁnd that
with increasing N, the time it takes for the peaks to de-
velop is reduced, as indicated by the vertical lines in the
ﬁgure. For N = 2 the resonant Rayleigh peak develops
before the second harmonic peak, while for N = 10 the
order is opposite. In addition, for N = 10 the SHG peak
transiently exceeds the Rayleigh peak also in magnitude.
To quantify these observations, we deﬁne T as the
time it takes a peak to reach half its maximum value.
As shown in Fig. 8, we ﬁnd that T as a function of N
shows a crossover around N = 2, from a regime where
the Rayleigh peak develops ﬁrst to a regime where the
SHG peak comes ﬁrst. Further, we ﬁnd for the Rayleigh
peak that the dependence of T on N is approximately
linear, while for the SHG peak is behaves as 1/N. For
the latter case the scaling is consistent with a superra-
diant behavior, where the duration of the superradiant
burst decreases as 1/N. In Fig. 8 we also show the num-
ber of photons P(ωb = 1) in the ﬂuorescent ﬁeld as a
function of N, and we ﬁnd it increases as N 2 for both
the Rayleigh and SHG peaks. Again this is consistent
with a superradiant mechanism.
Taken together, the results presented here indicate
that, compared to a single two-level system, the SHG
signal can be enhanced by considering an array of N of
two-level systems. Furthermore, the dependence on N
of both the emission time and intensity of the ﬁeld are
consistent with a superradiance behavior.


10
(a)
(b)
FIG. 7. Fluorescence spectra for an array with N two-level
systems, for N = 2 in panel (a) and N = 10 in panel (b).
The parameters are given by ωs = ωb = 1, ωa = 0.5, and
|α|2 = 9, and the bare couplings are taken as ga = 0.03 and
gb = 0.01. The white (red) vertical lines indicate the time
at which the height of the Rayleigh (SHG) peak reaches half
of its maximum value.
The spectral curves at such times
and at the end of the simulation time are explicitly shown
on the right side of the respective panels. The colormap is
normalized to the maximum value of the Rayleigh peak at
the ﬁnal time. The units of energy and time are respectively
given by ωs and ℏ/ωs.
FIG. 8.
The rise time T (dashed lines, crosses) and the
number of emitted photons P(ω) (solid lines, circles) of the
Rayleigh (blue) and second harmonic (red) peaks as a func-
tion of the number of two-level systems N. The parameters
are given by ωs = ωb = 1, ωa = 0.5, and |α|2 = 9, and the
bare couplings are taken as ga = 0.03 and gb = 0.01.
IV.
MOTION OF A TWO-LEVEL SYSTEM IN A
CAVITY
The two-level systems considered in the previous sec-
tions were at ﬁxed positions in space, like e.g. for a given
pair of levels in a quantum dot. However, if a two-level
system is meant to model a pair of atomic orbitals, then
the atomic center-of-mass motion can be an important,
if not crucial, element to take into account.
On the experimental side, compelling evidence comes
for example from studies of quantum control, where
atoms moving across an optical cavity provide informa-
tion about cavity photons [79], or laser beams across an
ion trap provide information about the internal state of
the ions [80]. On the theoretical side, the role of atom dy-
namics has been extensively considered [9, 21, 29, 32, 81–
90], often in terms of a generalized Jaynes-Cummings
model where the standard two-level, one-mode Hamilto-
nian is augmented by a kinetic energy operator (for the
center-of-mass motion).
Furthermore, the light-matter
coupling can become position dependent [9], for example
when the atom is moving inside/outside a optical cavity.
The solution of the generalized Jaynes-Cummings
model has been approached in many diﬀerent ways
[9, 21, 29, 32, 81–90].
For example, with or without
the RWA, with the center-of-mass motion described clas-
sically or quantum mechanically, using density-matrix
techniques or resorting to a direct solution of the gen-
eralized Rabi equations in wavefunction space.
Here,
we again consider the exact numerical time-evolution of
the full system wavefunction, thus avoiding the RWA.
Since we are interested in how the atom dynamics aﬀects
ﬂuorescence spectra, we consider a generalized Jaynes-
Cummings model with two (pump and ﬂuorescence)
modes, and with a center-of-mass that moves longitudi-
nally across a cavity of ﬁnite length. Transverse motion
is not considered (i.e., space-wise, our system is strictly
one-dimensional).
We treat the atom dynamics quan-
tum mechanically, but for comparison we also consider
a classical description via the Ehrenfest approximation.
Our approach includes all these element on equal-footing,
and in a single coherent description. This can oﬀer an
advantage: for example, the cavity boundaries can have
nontrivial eﬀects on the spectra which depend also on the
level of description.
Since we intend to look only in a preliminary and
explorative way at ﬂuorescence spectra in this setup,
we already here anticipate that in our calculations the
“atomic” mass value is taken rather small, but not so
small that it is necessary to take into account spatial dis-
persion eﬀects in the radiation-matter interaction. The
purpose of this choice is twofold: on the one hand, the
atom moves “faster”, which alleviates the costs of the nu-
merical time evolution to reach the long-time limit. On
the other hand the role of quantum eﬀects in the nuclear
motion is enhanced, since on increasing the value of the
atomic mass a classical description becomes increasingly
appropriate.


11
It is worth to mention that excitons in solid-state sys-
tems (e.g.
heterostructures [91]) can also be used for
two-level atom optics in quantized light ﬁelds, with the
exciton dynamics manipulated by optical means. This
option has the merit that the value of the exciton elec-
tron/hole eﬀective masses (and thus of the total mass)
can be tailored by manipulating the band-edge curva-
tures. Furthermore, using a mass-scaling transformation
as described in Appendix A 3, the calculations and results
to follow can qualitatively relate to microwave transitions
of atoms with realistic masses.
Out of this discussion, the Hamiltonian to consider is
ˆH(t) = ˆp2
2M + ϵ1ˆc†
1ˆc1 + ϵ2ˆc†
2ˆc2 + ωaˆa†ˆa + ωbˆb†ˆb
(27)
+
h
ga(ˆx, t)(ˆa† + ˆa) + gb(ˆx, t)(ˆb† + ˆb)
i
(ˆc†
1ˆc2 + ˆc†
2ˆc1)
where ˆp and ˆx are the momentum and position operators
of the atomic center of mass, M is the atomic mass. As
mentioned above, Eq. (27) satisﬁes a useful scaling prop-
erty (see Appendix A 3.) For the deﬁnition of the other
quantities, we refer to Eqs. (1-4). As before, we assume
that the atom is occupied by a single spinless electron,
that the cavity ﬁeld is of frequency ωa and in a coherent
state deﬁned by ˆa|α⟩= α|α⟩. Further, the ﬂuorescent
ﬁeld is of frequency ωb and initially in the vacuum state
ˆb|0⟩= 0.
The interaction between the light and the atom is given
by the couplings ga(ˆx, t) and gb(ˆx, t) respectively, with
the spatial dependence of the coupling ga coming from
the spatial proﬁle of the cavity mode. We assume that the
atomic motion happens only along the cavity axis, and
restrict the length of the coordinate axis to the set X =
[0, L] of length L. We further divide the coordinate axis
into two parts Xin and Xout, corresponding respectively
to inside and outside the cavity, where the set Xin =
[x1, x2] is of length l = x2 −x1 and Xout = X \ Xin is
of length L −l. For consistency we need to take 0 <
x1 < x2 < L.
The cavity electric ﬁeld E is assumed
to be in the lowest mode, with a spatial proﬁle given
by E(x) = sin((x −x1)π/l) for x ∈Xin and E(x) = 0
otherwise. Using the characteristic function χI, which
is unity on the interval I and zero otherwise, the light-
matter couplings are then given by
ga(ˆx, t) = g sin
π(ˆx −x1)
l

χ
Xin (ˆx)
(28)
gb(ˆx, t) = g1e−Γ1tχ
Xin (ˆx) + g2e−Γ2tχ
Xout (ˆx).
(29)
For the coupling to the ﬂuorescent ﬁeld we have assumed
a constant coupling g1 (g2) inside (outside) the cavity,
with a phenomenological decay Γ1 (Γ2) taking into ac-
count collision eﬀects and an eﬀective coupling to addi-
tional radiation modes in the continuum.
A.
Time dependent ﬂuorescence spectrum
Taking into account the quantum motion of the center
of mass signiﬁcantly increases the numerical eﬀort nec-
essary to obtain the ﬂuorescence spectrum. To simplify
the calculations we therefore consider the ﬂuorescence
response in the one-photon limit, where the coupling be-
tween the ﬂuorescent ﬁeld and the atom only acts once
during the time evolution. For the initial state, it is as-
sumed that i) the atom is prepared with a nuclear wave-
function the form of which is
φ(x) = e−(x−x0)2/σ2eixp0,
(30)
and with the electron in level 1 (with energy ϵ1) ii) the
cavity ﬁeld is in a coherent state α, and there are zero
photons of the ﬂuorescent ﬁeld. Thus the system’s initial
state is denoted by |1, φ, α⟩(the label for ﬂuorescent state
being omitted, because of the zero-photon assumption).
The spectrum is deﬁned as the probability that at time
t there is one photon in the ﬂuorescent ﬁeld:
P(t, ω) =
X
ni
Z
dx |⟨i, x, n|ˆbe−i ˆ
Ht|1, φ, α⟩|2.
(31)
Here, the state |i, x, n⟩also contains zero photons of the
ﬂuorescent ﬁeld (note the ˆb operator immediately to the
right of ⟨i, x, n|). In the ﬁnal state, i denotes the elec-
tronic level (1 or 2), x the atomic position, n the number
of photons in the cavity ﬁeld, and P R
the trace over
i, x, n. The perturbative limit is obtained by assuming
that the time-evolution operator can be written as
e−i ˆ
Ht ≈
Z t
0
dt′ e−i ˆ
H0(t−t′) ˆH′(t′)e−i ˆ
H0t′
(32)
where ˆH′(t) = gb(ˆx, t)(ˆc†
1ˆc2 + ˆc†
2ˆc1)(ˆb† + ˆb) and ˆH0 =
ˆH −ˆH′. In Appendix C we show that in this limit the
probability is given by
P(t, ω) =
X
λ

X
λ′
e−i(ϵλ+ω)t −e−iϵλ′t−Γ1t
ω + ϵλ −ϵλ′ + iΓ1
S1
λλ′
(33)
+ e−i(ϵλ+ω)t −e−iϵλ′t−Γ2t
ω + ϵλ −ϵλ′ + iΓ2
S2
λλ′

⟨λ′|1, φ, α⟩

2
.
where ϵλ are the eigenenergies of ˆH0. The coeﬃcients
S1/2
λλ′ are given by
S1/2
λλ′ =
X
ni
Z
Xin/out
dx ⟨λ| ˆH′|i, x, n⟩⟨i, x, n|λ′⟩,
(34)
where λ, λ′ label complete sets of states of ˆH0 (but again
with zero photons in the ﬂuorescent ﬁeld), and Xin and
Xout have been deﬁned earlier. This expression, used in
the next section to calculate the ﬂuorescence spectrum,
is valid for all times t and has the practical advantage of
requiring only a single diagonalization of the Hamiltonian


12
(a)
(b)
(c)
(d)
FIG. 9.
Fluorescence spectra for classical and quantum
atomic motion.
Panel (a):
x(t) and p(t) for the classical
atomic dynamics within the Ehrenfest approximation. Panel
(b): nuclear probability density N(x, t) from the quantum dy-
namics. Panels (c) and (d) show snapshots of the correspond-
ing ﬂuorescence spectra P(ω) at diﬀerent times for classical
and quantum atomic dynamics respectively. Taking lc = 104
a.u., the initial conditions are given by p0 = 0.5 a.u., x0 = 4lc,
x1 = 4lc and x2 = 5lc in the classical case, and by p0 = 0.5
a.u., σ = 3lc, x0 = 3.5lc, x1 = 4lc, x2 = 5lc and L = 10lc
in the quantum case. The remaining parameters are M = 10
a.u., ϵ = ϵ2 −ϵ1 = 0.043 a.u., ωa = ϵ, α = 1, ga = 0.1ϵ,
g1 = 0.1ϵ, g2 = 0.01ϵ and Γ = 0.02ϵ.
ˆH0. However, in actual calculations the numerical grid
for the nuclear coordinate x is conﬁned to the interval
(0 ≤x ≤L), and, for a given L, the maximum useful t
is limited by the need to avoid wavepacket reﬂection at
the interval boundaries.
B.
Classical versus quantum nuclear motion
Before studying the full dynamics of the system, we
consider the classical limit of the nuclear dynamics as
given by the Ehrenfest approximation. Assuming that ˆx
and ˆp in Eq. (27) are replaced by classical variables x and
p evolving under the force F(t) = −⟨∂x ˆHt)⟩, the spatial
dependence of ga turns into a time-dependence through
ga(t) = ga(x(t)) = ga sin((x(t) −x1)π/L). For the cou-
pling to the ﬂuorescent ﬁeld we assume (temporarily)
that there is no decoherence in the cavity, so that
gb(t) = g1χ
Xin (x(t)) + g2e−Γ(t−t0)χ
Xout (x(t))
(35)
with t0 the time at which the atom has passed through
the cavity. The quantum evolution is obtained by solving
the time-dependent Schr¨odinger equation on a grid xn,
as described in more detail later on.
In the following
we work in atomic units (a.u.) and take M = 10 a.u.,
ϵ = ϵ2 −ϵ1 = 0.043 a.u., ωa = ϵ, α = 1, ga = 0.1ϵ, g1 =
0.1ϵ, g2 = 0.01ϵ and Γ = 0.02ϵ. Except for the coupling
g1 of the ﬂuorescent ﬁeld to the atom inside the cavity,
these values as the same as earlier in the paper with the
identiﬁcation ϵ = 0.043 a.u.. However, the value of g1
was increased to enhance the emission into the ﬂuorescent
ﬁeld, with ⟨ˆb†ˆb⟩≪1 still applying.
A physical notion of the chosen parameters can be
gathered by noting that for a cavity of length lc = λ/2
and ϵ = ℏωa/κ (for example, κ = 0.5 for SHG), we have
ϵ = (ℏcπ)/(κlc).
Choosing lc = 104 a.u., one obtains
ϵ ≈0.043/κ a.u., which can be a reasonable value for ex-
ample for excitons, and fairly consistent with the value
of M = 10 a.u. introduced above [92].
Within the given units, the spatial simulation interval
(cavity and outside) of our calculations is L = 10lc (i.e.
about 5µm), and the cavity boundaries are set at x1 = 4lc
and x2 = 5lc. As initial conditions we take p0 = 0.5 a.u.
and x0 = 4lc for the classical simulations, while in the
quantum simulations the initial wave packet is given by
the expression in Eq. (30) with p0 = 0.5 a.u., x0 = 3.5lc
and σ = 3lc. This momentum corresponds to a velocity
v ≈104 cm/s.
In Fig. 9 we compare the results obtained with the
Ehrenfest approximation with the results of the full quan-
tum evolution. To characterize the atomic motion, we
look in the classical case at the functions x(t) and p(t),
and in the quantum case at the nuclear probability den-
sity N(x, t) = P
in |⟨i, x, n|Ψ(t)⟩|2.
We see in Fig. 9 that in the classical case the atom
moves through the cavity with little resistance, and note
that the eﬀects oscillations of p is not visible in x due
to the scale of the ﬁgure. In contrast, the quantum re-
sults show a splitting of the atomic wave packet. This is
precisely the regime where the Ehrenfest approximation
fails [93], since in a classical description the atom must
be either reﬂected or transmitted. However, at t = 6400
a.u. the quantum particle has both a (larger) reﬂected
and (smaller) transmitted contribution, while the classi-
cal particle is out of the cavity already at t = 3000 a.u.
Thus the atom has a reduced amplitude in the barrier
and a reduced coupling to the ﬁeld. Compared to the
classical case, treating the atomic motion at the quan-
tum level also has a large impact on the ﬂuorescence
spectrum.
This is addressed in the bottom panels for
diﬀerent time snapshots (spectra at diﬀerent times are
rather similar to each other, and only the one at the lat-
est time is fully visible). The Ehrenfest result resembles
at great extent that of a stationary atom (cf. Fig. 2),
while the spectrum corresponding to the quantum mo-
tion looks qualitatively diﬀerent: It contains two main
peaks instead of four, and is asymmetric with respect to
the central frequency.
To understand this dissimilarity in behavior, we note
that in the classical approach the nuclear wavepacket is
perfectly localized both in position and momentum. By
contrast, the quantum amplitude gets smaller in the re-
pulsive barrier region. Further, the classical atom sees
only a single resonant frequency (Doppler shifted due to
the motion) and coupling to the cavity ﬁeld at each given


13
(a)
(b)
(c)
(d)
(e)
(f)
FIG. 10. Nuclear probability densities (top row) and atomic ﬂuorescence spectra (bottom row) for an atom moving through a
cavity of frequency ωa = ϵ2 −ϵ1 = 0.043 a.u. The vertical lines denote the boundaries of the cavity. The initial atomic state
is given by Eq. (30), and we take x0 = 3.5lc (except for panels (a) and (d) where x0 = 4.5lc) and σ = 3lc with lc = 104 a.u.
Panels (a) and (d) corresponds to p0 = 0, panels (b) and (e) to p0 = 0.5 a.u., and panels (c) and (f) to p0 = 2 a.u. The
remaining parameters are given by M = 10 a.u., ϵ = ϵ2 −ϵ1 = 0.043 a.u., α = 1, ga = 0.1ϵ, g1 = 0.1ϵ, g2 = 0.01ϵ, Γ1 = 0.01ϵ
and Γ2 = 0.02ϵ.
time of its travel through the cavity.
Since the ﬁeld
is strongest in the center of the cavity, where it takes
the same value as in the stationary case discussed above
(α = 1 and ga = 0.1ϵ), the main contribution to the
ﬂuorescence signal comes from when the atom is in this
region. However, compared to the stationary case there
is an enhancement of the spectrum for frequencies ωb ≈ϵ,
which most likely comes from ﬂuorescent photons emit-
ted in the regions where ga < 0.1ϵ and the fact that the
classical atom couples to the photons more strongly.
In contrast, the quantum atom simultaneously expe-
riences a range of resonance frequencies and ﬁeld-atom
couplings, there is a lot of structure in the corresponding
spectrum, and its wave function gets low within the bar-
rier region. From the shape of the nuclear wavepacket
we expect the dominant contribution to the ﬂuorescence
signal to come from when the atom is in the initial and ﬁ-
nal part of the cavity, the probability distribution being
mainly localized to these regions (see Fig. 9).
Conse-
quently the spectrum is closer to what could be expected
for a stationary atom weakly interacting with a light ﬁeld
(ga < 0.1ϵ), leading to a smaller splitting between the
Mollow peaks. However, a detailed explanation of the
asymmetric form of the spectrum is diﬃcult to give, but
plausibly related to the varying Doppler shifts associated
with the diﬀerent parts of the atomic wavepacket.
C.
Fluorescence and quantum motion
We now consider the ﬂuorescence spectra resulting
from a quantum evolution of the coupled atom-photon
system.
We take L = 10lc, and, as before, the cav-
ity is placed again between x1 = 4lc and x2 = 5lc.
To solve the Schr¨odinger equation we consider a grid
xn for the atomic position, with 500 points in the in-
terval [0, L].
The ﬂuorescence spectrum is computed
from Eq. (33), and to get the atomic probability density
N(x, t) = P
in |⟨i, x, n|Ψ(t)⟩|2 we solved the Schr¨odinger
equation without the ﬂuorescent ﬁeld.
We have veri-
ﬁed that the atomic dynamics is highly insensitive to the
presence of the ﬂuorescent ﬁeld, by explicitly solving the
Schr¨odinger equation with the complete Hamiltonian for
a number of values of the ﬂuorescence frequency. This
insensitivity is due to the weak atom-ﬁeld coupling and
the absence of the spatial dispersion eﬀects. The weak
coupling also guarantees that the ﬁrst order ﬂuorescence
spectrum is a good approximation of the exact one. In
the following we let M = 10 a.u. and ωa = 0.043 a.u.
be ﬁxed, and take ϵ = ωa or ϵ = 2ωa for the Mollow
or SHG regimes respectively. The remaining parameters
are α = 1, ga = 0.1ϵ, g1 = 0.1ϵ, g2 = 0.05ϵ, Γ1 = 0.01ϵ
and Γ2 = 0.02ϵ.
In Fig. 10 we show N(x, t) and P(ω, t) for ϵ = ωa. We
see that for a stationary atom placed in the center of the
cavity, x0 = 4.5lc and p0 = 0, the spectrum resembles the
Mollow spectrum in Fig. 2. We also ﬁnd that although


14
(a)
(b)
(c)
(d)
(e)
(f)
FIG. 11. Nuclear probability densities (top row) and atomic ﬂuorescence spectra (bottom row) for an atom moving through
a cavity of frequency ωa = (ϵ2 −ϵ1)/2 = 0.043 a.u. The vertical lines denote the boundaries of the cavity. The initial atomic
state is given by Eq. (30), and we take x0 = 3.5lc (except for panels (a) and (d) where x0 = 4.5lc) and σ = 3lc with lc = 104
a.u. Panels (a) and (d) corresponds to p0 = 0, panels (b) and (e) to p0 = 0.5 a.u., and panels (c) and (f) to p0 = 2 a.u. The
remaining parameters are given by M = 10 a.u., ϵ = ϵ2 −ϵ1 = 0.086 a.u., α = 1, ga = 0.1ϵ, g1 = 0.1ϵ, g2 = 0.01ϵ, Γ1 = 0.01ϵ
and Γ2 = 0.02ϵ.
the atomic wave packet is initially contained in the cavity,
parts of the probability distribution are ejected as time
progresses. For higher initial momentum p0, we see that
an atom initially outside the cavity (at x0 = 3.5lc) is
either split (for p0 = 0.5 a.u.)
or travels through the
cavity (for p0 = 2 a.u.). This is in agreement with the
expectation based on an atom moving in the presence
of a dipole force [94], where the force on the particle
is proportional to the negative of the detuning and the
gradient of the light intensity F ∼−[ωa−(ϵ2−ϵ1)]∂xI(x).
For a ﬁeld on resonance, an atomic motion in the positive
direction leads to a positive detuning via a Doppler shift,
so that atom is expelled from regions of higher intensity.
This is why a minimal non-zero momentum is needed to
pass through the cavity.
In Fig. 11 we show N(x, t) and P(ω, t) for ϵ = 2ωa.
We note that the results for N(x, t) are rather similar
to those in Fig. 10, presumably because the coupling to
the radiation is too weak to make a larger diﬀerence. As
for the Mollow regime above, we ﬁnd that for an atom at
rest with l x0 = 4.5lc and p0 = 0, the spectrum resembles
the stationary SHG spectrum in Fig. 4. For this initial
state the atomic probability distribution is trapped in the
cavity, consistent with motion under a dipole force as dis-
cussed above. For non-zero initial momentum p0, we ﬁnd
that the SHG signal is strongly suppressed, and that the
elastic scattering peak is broadened. Following consid-
erations similar to those for Fig. 10, this eﬀect is likely
ascribable to the ﬁnite extent of the atomic wavepacket.
Interestingly, and as for Fig. 10, on increasing p0 the in-
tensity of the SHG spectrum exhibits a non monotonic
behavior. Further, due to the emission of a ﬂuorescent
photon being delayed with respect to the atomic excita-
tion, the resonance frequency of the atom has time to
change slightly between the two events. In fact, being a
second order process, SHG is expected to be more sensi-
tive to this type of detuning than the resonant scattering.
Thus the combination of detuning and the decoherence
induced by emission from diﬀerent parts of the atomic
wavepacket is likely the cause of the suppression of the
SHG signal.
V.
CONCLUSIONS
In optics and in photonics, the two-level system plays
the role of a Rosetta-stone for light-matter interactions,
at the interface of quantum with classical and linear with
nonlinear behavior. In this work, we used this paradig-
matic system to address basic aspects of multi-photon ﬂu-
orescence in the time-dependent and stationary regimes.
The ﬂuorescence response was considered in three cases
of increasing complexity, namely in a single two-level sys-
tem interacting with an incident coherent ﬁeld, as well as
in an array of two-level systems, and a ﬁnally for a two-
level atom moving across a cavity.
By solving the Schr¨odinger equation through exact nu-
merical time-propagation, we showed that, depending


15
on the system and ﬁeld parameters, the time-dependent
ﬂuorescence spectrum develops distinct features that in
some cases are not captured by a semi-classical treatment
of the incident ﬁeld. Some of these features oﬀer direct
evidence that the usual selection rules of perturbative op-
tics, which consider photons and electrons separately, do
not apply in the strong coupling regime.
As clear-cut example, we showed that a second har-
monic signal (SHG) can occur in a two-level system. This
result was analyzed in terms of the parity of the coupled
electron-photon states, and we argued that this nonlinear
process is allowed even for parity invariant Hamiltonians.
We also studied the SHG process in a three-level system,
and showed that in an appropriate limit the three-level
system reproduces the results of the two-level system.
The SHG signal gets enhanced in a setup with N two-
level systems (compared to the N = 1 case), with trends
suggestive of superradiant behavior. This conclusion was
gathered by looking analytically at possible signatures of
a phase transition in the large N limit, and by analyzing
the onset of the SHG response in exact numerical calcu-
lations for N ≲10. However, compared to the case of
one atom at rest, both Mollow and SHG signals are sup-
pressed by atomic motion. This eﬀect is especially strong
in the quantum case: For quantum atomic motion, the
spread in space of the traveling nuclear wavepacket is
greatly increased by the presence of the barrier repre-
sented by the optical cavity, giving rise to a position de-
pendent Doppler shift. This in turn results in a large fre-
quency dispersion in the ﬂuorescence signal. It is impor-
tant to specify here that the noted quantum eﬀects were
enhanced by choosing artiﬁcially small nuclear masses.
However, although we focussed here on model atomic sys-
tems, it is a fair assumption that many of the results ob-
tained should carry over to solid state two-level systems,
where suitable (excitonic) masses could be engineered.
The aforementioned eﬀects are weak, and they mani-
fest at low intensities; this is conﬁrmed by the fact that,
for the chosen parameters, a linear and an all-orders
treatment of the ﬂuorescence ﬁeld provide an identical
scenario. This also means that our situation does not cor-
respond to standard heterodyne setups, and no intensity
renormalization is needed. Even so, said eﬀects should
be of some conceptual (if not practical) interest. It can in
fact be argued that the found superimposed Mollow-like
structure to the SHG signal is speciﬁcally distinctive of
the genuine “two-level” character of the material system,
as also gathered by looking at three-levels results. We
add that a similar (albeit weaker) superimposed struc-
ture also occurs for higher-order harmonics.
Concerning dissipation eﬀects, we expect on purely
speculative grounds that the peculiar four-peak structure
of the Mollow spectrum could be dimmed, while the SHG
signal would considerably change but still survive for not
too strong dissipation.
In conclusion, we have addressed general features of
multi-photon ﬂuorescence, but only in very simple model
systems: The inclusion of additional radiation modes,
more general time-dependent couplings, a careful inclu-
sion of bath eﬀects, and more realistic atomic and solid-
state setups are possible directions for future work, and
to conﬁrm in a broader sense the robustness of present re-
sults. Ultimately, true validation comes from experiment,
and we hope that our work will stimulate investigations
in that direction.
ACKNOWLEDGMENTS
We wish to thank W. P. Schleich for useful discussions.
E. V. B was supported by Crafoordska Stiftelsen. C. V.
was supported by the Swedish Research Council.
Appendix A
We provide here some additional motivation and detail
about the model and the method of solution.
1.
About dissipative eﬀects
For the ﬁrst two typologies of systems considered, i.e.
one or many two-level system(s) at rest, we assume that
(via e.g.
a cavity-geometry or an high-optical quality
sample in ultra-high vacuum and at helium temperature),
the inhomogeneous broadening has been made as neg-
ligible as possible.
In this way, the focus is solely on
the homogeneous broadening. This is due to both radia-
tive and non-radiative components, that in our treatment
are accounted for by a total phenomenological Lorentzian
damping.
For the the third typology of system (i.e. the “atom”
in motion, and where the dissipative environment can
be considerably diﬀerent) we have followed a common
practice in the literature, considering only the moving
material system and the relevant modes of interest. Ul-
timately, this choice is also dictated by computational
convenience, since the full quantum treatment of nuclear
motion adds considerable complexity to the numerics.
Looking ahead, a possible way to include dissipative
eﬀects is via Lindblad-type master equations or, alter-
natively, via non-equilibrium Green’s functions (NEGF).
NEGF permit to include memory eﬀects and the disper-
sive contribution of the environment in a very direct and
systematic way, and it would be rather interesting to per-
form a comparison between NEGF and master equation
results. These calculations and comparisons are deferred
to future work.
2.
Computational details
The short iterated Lanczos method is an eﬃcient al-
gorithm to approximate the time-evolution operator U.


16
This is done by constructing U in a small optimized sub-
space (the Krylov space), which allows to maintain uni-
tarity of U (in contrast to a straightforward Taylor ex-
pansion) while being numerically eﬃcient. We used this
algorithm to propagate the many-particle Schr¨odinger
equation, and additional details can be found in Ref. [58].
Regarding the choice of basis, we use two basis states
|1⟩and |2⟩two describe the “atomic” electron states,
and the number states |na⟩and |nb⟩for the coherent
and ﬂuorescent ﬁelds respectively. In the last part of the
manuscript, where we study the motion through a cavity,
we use the position basis |xn⟩on an equidistant grid to
describe the “atomic” center of mass.
Finally, the choice of cut-oﬀnumber(s) for the radi-
ation modes is determined by the convergence (i.e. by
increasing the number of states until the results are con-
verged within machine accuracy).
For the ﬂuorescent
ﬁeld we found it was suﬃcient in all cases considered
to use a maximum nb ≈10. For the coherent ﬁeld the
numerical cut-oﬀdepends on α, since the coherent ﬁeld
follows a Poissonian distribution when written in terms
of number states. For α = 1 and α = 5 respectively, we
found that a cut-oﬀat na ≈30 and na ≈150 is enough.
3.
A scaling property
The Hamiltonian in Eq. (27) satisﬁes a scaling prop-
erty relating the full quantum dynamics of systems with
diﬀerent masses.
Speciﬁcally, we start by considering
a scaling parameter Z, and the Schr¨odinger equation
i∂tψ(t) = ˆH(t)ψ(t).
Dividing by Z, and setting t′ =
Zt, we get i∂t′φ(t′) = Z−1 ˆH(t′/Z)φ(t′), where φ(t′) =
ψ(t′/Z). By relabeling the time variable, t′ →t, we then
have i∂tφ(t) = ˜H(t)φ(t), where ˜H(t) = Z−1 ˆH(t/Z). Ac-
cording to this scaling prescription, a given numerical
calculation represents in fact a entire one-parameter set
of numerical simulations, where the integration interval,
the time dependence in ˆH and the fermion-boson inter-
actions are changed.
Appendix B
We consider here the adiabatic elimination approxima-
tion (AEA) for the three-level Hamiltonian given by the
sum of Eqs. (9),(10) and the free ﬁeld part. Most often,
the AEA is done in connection with the rotating wave
approximation (RWA) (see e.g. [64–66, 95]), and choos-
ing the order in which AEA and RWA are performed can
be important [95]. Since our considerations here aim to
be qualitative and general in character, we use for sim-
plicity the more common protocol where the RWA is in-
troduced before the AEA [65], and before the pump ﬁeld
undergoes a transformation to a coherent photon picture.
Proceeding in this way, we obtain
ˆHRW A = f(t)[ˆc†
3ˆc2 + ˆc†
2ˆc1]ˆa + gb(t)ˆc†
3ˆc1ˆb + H.c.
+ ϵ1ˆc†
1ˆc1 + ϵ2ˆc†
2ˆc2 + ϵ3ˆc†
3ˆc3 + ωaˆa†ˆa + ωbˆb†ˆb
(B1)
After an AEA of the intermediate level |2⟩(therefore,
the dynamical Stark eﬀect due to the intermediate level
is neglected) we get
ˆHAEA
RW A = ˆc†
3ˆc1

f(t)ˆa2 + gb(t)ˆb

+ H.c.
+ ϵ3 −ϵ1
2
(ˆc†
3ˆc3 −ˆc†
1ˆc1) + ωaˆa†ˆa + ωbˆb†ˆb.
(B2)
At this point the coherent state picture could be
introduced.
However, already at this stage, the AEA
two-level model of Eq. (B2) seems rather diﬀerent from
the original two-level model of Eqs. (1)-(4). Therefore,
the results of the main text for the SHG and Mollow
spectra in a two-level system should not be ascribed to
an adiabatic suppression of the virtual level.
Appendix C
We want to calculate the probability deﬁned in
Eq. (31) and repeated here for convenience:
P(t, ω) =
X
ni
Z
dx |⟨i, x, n|ˆbe−i ˆ
Ht|1, φ, α⟩|2.
(C1)
In the perturbative limit, the time-evolution operator be-
comes:
e−i ˆ
Ht ≈
Z t
0
dt′ e−i ˆ
H0(t−t′) ˆH′(t′)e−i ˆ
H0t′,
(C2)
and to further simplify the analysis, we deﬁne the prob-
ability amplitude
Ax
in(t, ω) = −iθ(t)
Z t
0
dt′ ⟨i, x, n|ˆbe−i ˆ
H0(t−t′) ˆH′
(C3)
× e−i ˆ
H0t′|1, φ, α⟩.
Since ˆH0 is independent of time, the probability ampli-
tude can be found through a straightforward expansion
in the eigenstates of ˆH0. In the expression for the prob-
ability P above, we trace over a complete set of ﬁnal
states |i, x, n⟩, but since any complete set is allowed we can instead choose to trace over the eigenstates of ˆH0. In the


17
following we therefore consider the probability amplitude Aλ(t, ω), and by inserting a set of complete states we ﬁnd
⟨λ|ˆbe−i ˆ
H0(t−t′) ˆH′(t′)e−i ˆ
H0t′|1, φ, α⟩=
X
λ′′λ′
⟨λ|ˆb|λ′′⟩e−i(ϵλ′′+ω)(t−t′)H′
λ′′λ′(t′)e−iϵλ′t′⟨λ′|1, φ, α⟩
=
X
λ′
e−i(ϵλ+ω)(t−t′)−iϵλ′t′H′
λλ′(t′)⟨λ′|1, φ, α⟩.
(C4)
Now the matrix elements H′
λλ′(t′) can be broken into two parts according to
H′
λλ′(t′) =
X
ni
Z
dx ⟨λ| ˆH′(t′)|i, x, n⟩⟨i, x, n|λ′⟩
(C5)
=
X
ni
Z
Xin
dx g1e−Γ1t′⟨λ| ˆH′|i, x, n⟩⟨i, x, n|λ′⟩+
X
ni
Z
Xout
dx g2e−Γ2t′⟨λ| ˆH′|i, x, n⟩⟨i, x, n|λ′⟩
= g1e−Γ1t′S1
λλ′ + g2e−Γ2t′S2
λλ′
where the coeﬃcients S1
λλ′ and S2
λλ′ are given by
S1
λλ′ =
X
ni
Z
Xin
dx ⟨λ| ˆH′|i, x, n⟩⟨i, x, n|λ′⟩
S2
λλ′ =
X
ni
Z
Xout
dx ⟨λ| ˆH′|i, x, n⟩⟨i, x, n|λ′⟩.
(C6)
Integrating over t′ we ﬁnd the probability amplitude to be
Aλ(t, ω) =
X
λ′
e−i(ϵλ+ω)t −e−iϵλ′t−Γ1t
ω + ϵλ −ϵλ′ + iΓ1
S1
λλ′ + e−i(ϵλ+ω)t −e−iϵλ′t−Γ2t
ω + ϵλ −ϵλ′ + iΓ2
S2
λλ′

⟨λ′|1, φ, α⟩,
(C7)
and inserting this into the expression for the probability we ﬁnd
P(t, ω) =
X
λ

X
λ′
e−i(ϵλ+ω)t −e−iϵλ′t−Γ1t
ω + ϵλ −ϵλ′ + iΓ1
S1
λλ′ + e−i(ϵλ+ω)t −e−iϵλ′t−Γ2t
ω + ϵλ −ϵλ′ + iΓ2
S2
λλ′

⟨λ′|1, φ, α⟩

2
.
(C8)
If necessary, it is possible to further simplify Eq. (C8) by going at long times (i.e. where the exponentials e−Γkt tend
to zero) provided that L is correspondingly taken large enough to avoid atom reﬂection at the ends of the x-coordinate
domain. Arguing that the cross terms vanish for t →∞, the asymptotic limit becomes
P(ω) =
X
λ

X
λ′

1
ω + ϵλ −ϵλ′ + iΓ1
S1
λλ′ +
1
ω + ϵλ −ϵλ′ + iΓ2
S2
λλ′

⟨λ′|1, φ, α⟩

2
.
This latter result makes contact with the long time limit of the static-atom case of Sec. II. However, to calculate the
time-dependent ﬂuorescence spectrum for a moving atom we will go back to the full expression for P(t, ω) in Eq. (C8)
here, or Eq. (33) in the main text.
[1] A. Einstein, Z. Phys. 18, 121 (1917).
[2] P. A. M. Dirac, The Quantum Theory of the Emission
and Absorption of Radiation, Proc. Roy. Soc. A 114, 243.
(1927); doi:10.1098/rspa.1927.0039.
[3] V. Weisskopf, and E. Wigner, Z. Phys. 63, 54 (1930).
[4] M. Gabriela Lagorio, G. B. Cordon and A. Iriel, Re-
viewing the relevance of ﬂuorescence in biological sys-
tems, Photochem. Photobiol. Sci. 14, 1538 (2015); DOI:
10.1039/c5pp00122f.
[5] S. Neppl, R. Ernstorfer, A. L. Cavalieri, C. Lemell, G.
Wachter, E. Magerl, E. M. Bothschafter, M. Jobst, M.
Hofstetter, U. Kleineberg, J. V. Barth, D. Menzel, J.
Burgd¨orfer, P. Feulner, F. Krausz and R. Kienberger,
Nature 517, 342 (2015).
[6] S. H. Chew, K. Pearce, C. Sp¨ath, A. Guggenmos, J.
Schmidt, F. S¨ußmann, M. F. Kling, U. Kleineberg, E.


18
M˚arsell, C. L. Arnold, E. Lorek, P. Rudawski, C. Guo,
M. Miranda, F. Ardana, J. Mauritsson, A. L’Huillier,
A. Mikkelsen, Wiley-Blackwell, Attosecond Nanophysics
Wiley-Blackwell 325 (2015).
[7] L. Allen and J. H. Eberly, Optical resonance and two-level
atoms (Wiley, 1975).
[8] B. W. Shore, The Theory of Coherent Atomic Excitation
(Wiley-Interscience, New York, 1990).
[9] W. P. Schleich, Quantum optics in phase space (Wiley-
VCH, Berlin, 2001).
[10] I. I. Rabi, S. Millman, P. Kusch and J. R. Zacharias,
Phys. Rev. 55, 526 (1939).
[11] F. Bloch and A. Siegert, Phys. Rev. 57, 522 (1940).
[12] S. Stenholm,Saturation eﬀects in RF spectroscopy. I.
General theory J. Phys. B. 5, 878 (1972); ibid. 890 (1972).
[13] E. T. Jaynes and F. W. Cummings, Proc. IEEE 51, 89
(1963).
[14] H. J. Carmichael and D. F. Walls, J. Phys. B 9, 1199
(1976).
[15] Bruce W. Shore and Peter L. Knight, The Jaynes-
Cummings Model, Journal of Modern Optics, 40, 1195
(1993), DOI: 10.1080/09500349314551321
[16] K. Fujii, Dynamics of an N-level system of atoms inter-
acting with laser ﬁelds, J. Math. Sci., 153, 57 (2008)
[17] Q. Xie, H. Zhong, M. T. Batchelor and C. Lee, J. Phys.
A: Math. Theor. 50, 113001 (2017).
[18] J. H. Eberly, N. B. Narozhny, and J. J. Sanchez-
Mondragon, Phys. Rev. Lett. 44, 1323 (1980).
[19] S. Gu´erin, F. Monti, J.-M. Dupont and H. R. Jauslin, J.
Phys. A: Math. Gen. 30, 7193 (1997).
[20] A. F. Kockum, A. Miranowicz, S. De Liberato, S.
Savasta, and F. Nori, Ultrastrong coupling between light
and matter Nature Reviews Physics 1, 19 (2019).
[21] C. Cohen-Tannoudji and D. Gu´ery-Odelin, Advances in
atomic physics: An overview (World Scientiﬁc 2011).
[22] B. R. Mollow, Phys. Rev. 188, 1969 (1969).
[23] S. Swain, J. Phys. A: Math. Nucl. Gen. 6 192 (1973);
ibid., 1919 (1973).
[24] A. D’Andrea, Phys. Rev. A 39, 5143 (1989).
[25] C. H. Keitel, Vacuum modiﬁed resonance ﬂuorescence in
intense laser ﬁelds, J. Mod. Opt. 43, 1555 (1996); DOI:
10.1080/09500349608232827
[26] D. E. Browne and C. H. Keitel, Resonance ﬂuorescence in
intense laser ﬁelds, J. Mod. Opt. 47, 1307 (2000); DOI:
10.1080/09500340008235106
[27] E. Perfetto and G. Stefanucci, Some exact properties of
the nonequilibrium response function for transient pho-
toabsorption, Phys. Rev. A 91, 033416 (2015).
[28] Q. Xu, Sideband structure spontaneous spectrum without
the rotating wave approximation, Eur. Phys. J. D 51, 387
(2009); DOI: 10.1140/epjd/e2008-00235-7
[29] M. O. Scully, V. V. Kocharovsky, A. Belyanin, E. Fry,
and F. Capasso, Phys. Rev. Lett. 91, 243004 (2003).
[30] C. O’Brien and M. O. Scully, J. Mod. Opt, 63, 27 (2015).
[31] Semi-classical and quantum Rabi models: in celebration
of 80 years, J. Phys. A: Math. Theor. (2017).
[32] H. Walther, B. T. H, Varcoe, B.-G. Englert and T.
Becker, Rep. Prog. Phys. 69, 1325 (2006).
[33] Quantum Control of Light and Matter, Optics Comm.
264, 247 (2006).
[34] F. A. Wolf, F. Vallone, G. Romero, M. Kollar, E. Solano,
and D. Braak, Dynamical correlation functions and the
quantum Rabi model, Phys. Rev. A 87, 023835 (2013).
[35] F. A. Wolf, M. Kollar, and D. Braak, Phys. Rev. A 85,
053817 (2012).
[36] H.-P. Eckle and H. Johannesson, J. Phys. A: Math.
Theor. 50, 294004 (2017).
[37] Free-Space Quantum Electrodynamics with a Single Ry-
dberg Superatom, Asaf Paris-Mandoki, Christoph Braun,
Jan Kumlin, Christoph Tresp, Ivan Mirgorodskiy, Flo-
rian Christaller, Hans Peter B¨uchler, and Sebastian Hof-
fer, Phys. Rev. X 7, 041010 (2017).
[38] C. S´anchez Mun˜oz, A. Frisk Kockum, A. Miranowicz,
and F. Nori, Ultrastrong-coupling eﬀects induced by a
single classical drive in Jaynes-Cummings-type systems,
arXiv:1910.12875v1.
[39] Shi-Yao Zhu, L. M. Narducci, and M. O. Scully, Phys.
Rev. A 52, 4791 (1995).
[40] M. Malekakhlagh and A. W. Rodriguez, Phys. Rev. Lett.
122, 043601 (2019).
[41] Wang Ning, Gong Zhi-Rui, Lu Jing, Zhou Lan, Frontiers
in Physics 7, 127 (2019).
[42] M. Stobi´nska, G. Alber, and G. Leuchs, Europhys. Lett.
86, 14007 (2009).
[43] L. Ortiz-Guti´errez, R. C. Teixeira, A. Eloy, D. Ferreira
da Silva, R. Kaiser, R. Bachelard, and M. Fouch´e, New
J. Phys. 21, 093019 (2019).
[44] C. Guerlin, J. Bernu, S. Delglise, C. Sayrin, S. Gleyzes, S.
Kuhr, M. Brune, J.-M. Raimond and S. Haroche, Nature
448, 889 (2007).
[45] S. Gleyzes, S. Kuhr, C. Guerlin, J. Bernu, S. Delglise,
U. B. Hoﬀ, M. Brune, J.-M. Raimond and S. Haroche,
Nature 446, 297 (2007).
[46] K. W. Murch, S. J. Weber, K. M. Beck, E. Ginossar and
I. Siddiqi, Nature 499, 62 (2013).
[47] L.Pilozzi, A. D’Andrea and K. Cho, Phys. Rev. B 76,
245312 (2007).
[48] A. Joshi and S. V. Lawande, Phys. Rev. A 42, 1752
(1990).
[49] A. Joshi, Phys. Rev. A 58, 4662 (1998).
[50] K. E. Dorfman and S. Mukamel, PNAS 115, 1451 (2018).
[51] M. Cini, A. D’Andrea, and C. Verdozzi, Phys. Lett.
A180, 430 (1993).
[52] M. Cini, A. D’Andrea, and C. Verdozzi, Int. Jour. Mod.
Phys. B9, 1185 (1995).
[53] M. Cini and A. D’Andrea, J. Phys. C: Solid State Phys.
21, 193 (1988)
[54] C. Cohen-Tannoudji, J. Dupont-Roc, and G. Grynberg,
Atom-Photon interactions: Basic processes and applica-
tions, Wiley-VCH (1992).
[55] With the chosen units, the parameters values employed
in our model calculation, while generic in character, fall
well within the the physical range of validity.
[56] P. L. Knight and L. Allen, Concepts of quantum optics,
Pergamon (1983).
[57] It might be worth to add that for exciton-polaritonics
in 3D crystals, another class of template systems where
our results might be of potential relevance, the sponta-
neous decay occurs because of the ﬂuctuations in the po-
laritonic vacuum. Furthermore, being described by cou-
pled Schr¨odinger and Maxwell equations, the polaritonic
eigenfunction does not spontaneously decay in the bulk,
but rather at the solid/vacuum interface.
[58] T. J. Park and J. C. Light, J. Chem. Phys. 85, 5870
(1986).
[59] E. Bostr¨om, A. Mikkelsen, and C. Verdozzi, Phys. Rev.
B 93, 195416 (2016).
[60] J. H. Eberly and K. W´odkiewicz, J. Opt. Soc. Am., 67,


19
1252 (1977).
[61] E. del Valle, A. Gonzalez-Tudela, F. P. Laussy, C. Teje-
dor, and M. J. Hartmann, Phys. Rev. Lett. 109, 183601
(2012).
[62] Z. S. Yang, N. H. Kwong, R. Binder and A. L. Smirl, J.
Opt. Soc. Am. B 22, 2144 (2005).
[63] A. Ridolfo, M. Leib, S. Savasta, and M. J. Hartmann,
Phys. Rev. Lett. 109, 193602 (2012).
[64] P. Alsing and M. S. Zubairy, J. Opt. Soc. Am. B 4, 177
(1987).
[65] S.-C. Gou, Phys. Rev. A 40, 5116, (1989).
[66] E. Brion, L. H. Pedersen and K. Mølmer, J. Phys. A:
Math. Theor. 40, 1033 (2007).
[67] The system in Eqs. (1-4) should not be confused with the
eﬀective two-level one obtained from a three level system
via adiabatic elimination, as described in appendix B.
[68] S. He, C. Wang, Q.-H. Chen, X.-Z. Ren, T. Liu, and
K.-L. Wang, Phys. Rev. A 86 033837 (2012).
[69] R. H. Dicke, Phys.Rev. 93, 99 (1954).
[70] R. Bonifacio and L. A. Lugiato, Phys. Rev. A 11, 1507
(1975).
[71] R. Bonifacio, P. Schwendimann, and F. Haake, Phys.
Rev. A 4, 302 (1971); ibid. 854 (1971).
[72] M. Gross and S. Haroche, Phys. Rep. 93, 301 (1982).
[73] C. Leonardi, F. Persico, and G. Vetri, Riv. Nuovo Cim.
9, 1 (1986).
[74] P.
Schwendimann
and
A.
Quattropani,
Quantum-
statistical properties of the emission from two-level atoms
embedded in coupled cavities, Phys. Rev. A 86, 043811
(2012). See also P. Schwendimann and A. Quattropani,
arXiv:1311.3848.
[75] M. Tavis and F. W. Cummings, Phys. Rev. 170, 379
(1968).
[76] P. Kirton, M. M. Roses, J. Keeling, and E. G. Dalla
Torre, Adv. Quant. Tech. 2, 1970013 (2019).
[77] T. Ikawa and K. Cho, Fate of the superradiant mode in a
resonant Bragg reﬂector, Phys. Rev. B 66, 085338 (2002).
[78] T. Holstein and H. Primakoﬀ, Phys. Rev. 58, 1098
(1940).
[79] M. Brune, F. Schmidt-Kaler, A. Maali, J. Dreyer, E. Ha-
gley, J-M. Raimond and S. Haroche, Phys.Rev.Lett. 76,
1800 (1996).
[80] D.M. Meekhof, C. Monroe, B. E. King, W. M. Itano and
D. J. Wineland, Phys.Rev.Lett. 76, 1796 (1996).
[81] P. Meystre, E. Schumacher, and S. Stenholm, Opt.
Comm. 73, 443 (1989).
[82] A. P. Kazantsev, G. I. Surdutovich, and V. P. Yakovlev,
Mechanical Action of Light on Atoms (World Scientiﬁc
1990).
[83] M. J. Holland, D. F. Walls, and P. Zoller, Phys.Rev. Lett.
67, 1716 (1991).
[84] J. I. Cirac, R. Blatt, A. S. Parkins, and P. Zoller, Phys.
Rev. A 48, 2169 (1993).
[85] G.M. Meyer, M. O. Scully, and H. Walther, Phys. Rev.
A 56, 4142 (1997).
[86] E. V. Goldstein and P. Meystre, Phys.Rev. A 56, 5135
(1997).
[87] M. L¨oﬄer, G. M. Meyer, M. Schr¨oder, M. O. Scully, and
H. Walther, Phys. Rev. A 56, 4153 (1997).
[88] M. Schr¨oder, K. Vogel, W. P. Schleich, M. O. Scully, and
H. Walther, Phys. Rev. A 56, 4164 (1997).
[89] P. Meystre, Atom Optics (Springer, 2001).
[90] F. Krumm and W. Vogel, Phys. Rev. A 97, 043806
(2018).
[91] A. Stahl and I. Balslev, Electrodynamics of the Semicon-
ductor Band Edge (Springer, 1987).
[92] Caution should be exerted in associating too closely our
parameter values to the case of excitons: inside the cav-
ity, it is more appropriate to consider a polaritonic pic-
ture. Thus, the reference to excitons rather aims to show-
ing that the parameters of our model study are plausible
in the physical range, rather than to connecting to a con-
crete and speciﬁc case.
[93] E. Bostr¨om, M. Hopjan, A. Kartsev, C. Verdozzi and C.-
O. Almbladh, J. Phys.: Conf. Ser. 696, 012007 (2016).
[94] G. Grynberg, A. Aspect, and C. Fabre, Introduction to
Quantum Optics, Cambridge University Press (2010).
[95] M. P. Fewell, Opt. Commun. 243, 125 (2005).
