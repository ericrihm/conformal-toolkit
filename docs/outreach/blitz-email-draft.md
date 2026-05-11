**Subject:** Computational toolkit implementing your conformal hypersurface invariant classification

Hi Sam,

I'm Eric Rihm, a software engineer working at the intersection of differential geometry and computation. I've been following your work on conformal hypersurface invariants -- particularly arXiv:2511.02072 and your conformal fundamental forms paper in J. Math. Phys. -- and I wanted to reach out because I've built something I think you'd find interesting.

**What I built:** A SageMath toolkit that computes your conformal hypersurface invariant classification as working software -- the first implementation, as far as I know. The toolkit symbolically computes L_1 and L_2, enumerates invariants by conformal weight and ambient dimension, and derives Willmore densities W_2 and W_4 directly from the conformal fundamental forms. It also includes a tractor calculus module that could connect to tractor-based constructions of hypersurface invariants. The whole thing sits on top of a symbolic Riemannian geometry engine with ambient metric, curvature tensor, and Schouten tensor computation built in.

**Why this matters for your program:** Your classification organizes the space of conformal hypersurface invariants theoretically -- this toolkit lets you explore that space computationally. You can enumerate invariants at a given weight, verify identities symbolically, and push into regimes (higher-order L_k, higher ambient dimension) where hand computation becomes impractical. Notebook 02 in the repo walks through your classification explicitly.

**Concrete collaboration ideas:**

1. **JOSS paper** -- A toolkit paper for the Journal of Open Source Software, with you as co-author. Your mathematical framework, my implementation. Establishes the software as a citable resource for the conformal geometry community.
2. **Research paper** -- Extend the classification computationally to higher-order L_k and use the toolkit to discover new identities or verify conjectured ones. The software can search spaces that are infeasible by hand.
3. **ML/applications paper** -- I've also built a pipeline that uses conformal invariants as features for graph neural networks, applied to shape analysis. This is a novel bridge between conformal geometry and geometric deep learning -- could be a good fit for an ICML or NeurIPS workshop.

The repo is public with 125 tests and 6 example notebooks:
https://github.com/ericrihm/conformal-toolkit

Notebook 02 directly demonstrates your classification and would be the best place to start.

I'd welcome the chance to discuss any of these directions -- happy to set up a call or answer questions over email. No pressure on timeline; I know you're busy.

Best,
Eric Rihm
admin@strata-networks.com
