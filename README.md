## Semantic Correction for Pre-Trained Word Embeddings
### by Tanmay Shah, Lukas Geiger, Carrie Hay Kellar

This repo contains code to inject semantic information into pre-trained syntactic word embeddings by balancing syntactic and semantic loss. 

The syntactic loss is calculated from pre-trained embeddings and assumed to be zero on initialization, since pre-trained embeddings are trained on syntactic features ( e.g. context windows or dependency relations). 

The semantic loss is calculated using information from the semantic_info folder. This contains files of word pairs categorized by relation. We use the cosine distance between words in a pair as a metric of semantic loss; the lower the cosine similarity, the higher the loss. Semantic information includes information from synonyms, antonyms, hypernyms, hyponyms, and meronyms.

We find that using this post-processing method improves performance across a range of tasks, such as word similarity, concept categorization, and word analogy.

### Organization of Files:
\begin{enumerate}
    \item truncateVocab.ipynb: This file trims the vocabulary of pre-trained embeddings down to specified size.
    \item initEmbeddings.ipynb: This file initializes 3 types of embeddings: Random, Average, and PMI-based.
    \item semanticCorrection.ipynb: Implements the semantic correction procedure.
    \item evalEmbeddings.ipynb: Runs evaluation of embeddings on standard benchmarks as well as the Better Analogy Test Set (BATS).
\end{enumerate}

### SynGCN Directory:

The original idea for this project was to implement a semantic regularizer on graph convolutional networks. However, we were not able to achive this due to limited time and compute resources. However, the SynGCN folder contains a full, working implementation of the proposed idea. The implementation takes as input a corpus (Wikipedia, here), generates batches, implements Syntacitic GCN and trains the model. 

While this repo was forked from the paper that describes the original idea (https://github.com/malllabiisc/WordGCN), some implementation issues and redundancies were found and fixed. Most importantly, redundancies in batch generation, adjacency matrix calculation, and model definition were addressed. Furthermore, the whole implementation is updated from TensorFlow v1.x to TensorFlow v2.x and comprehensively simplified.

While results of the \textit{SynGCN + Semantic Regularization} implementation were unable to be fully-processed, from the results of small-scale experiments, there is cause to believe that semantic regularization while training word embeddings can lead to significant improvements in tasks like word similarity, concept categorization, and word analogy.

We hope to train and explore this implementation in the future.

### Directory Tree

