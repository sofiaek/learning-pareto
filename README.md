# Learning Pareto-Efficient Decisions with Confidence
This repository contains code to replicate the experimental results in:

Sofia Ek, Dave Zachariah, Petre Stoica. ["Learning Pareto-Efficient Decisions with Confidence".](https://arxiv.org/pdf/2110.09864) 2021.

## Abstract 
The paper considers the problem of multi-objective decision support when outcomes are uncertain. 
We extend the concept of Pareto-efficient decisions to take into account the uncertainty of decision outcomes across varying contexts. 
This enables quantifying trade-offs between decisions in terms of tail outcomes that are relevant in safety-critical applications. 
We propose a method for learning efficient decisions with statistical confidence, building on results from the conformal prediction literature. 
The method adapts to weak or nonexistent context covariate overlap and its statistical guarantees are evaluated using both synthetic and real data. 

## Results
The files named weighted_split_CQR_xxx replicates all the results in the paper.

The conformal prediction part of the implementation is based on code from https://github.com/yromano/cqr, 
but extended to account for a distributional shift that arise between the training and interventional distributions.

The STAR dataset used in some of the numerical experiments is available here:

C.M. Achilles, Helen Pate Bain, Fred Bellott, Jayne Boyd-Zaharias, Jeremy Finn, John Folger, John Johnston, Elizabeth Word, , ["Tennessee's Student Teacher Achievement Ratio (STAR) project".](https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:1902.1/10766) 2008.
