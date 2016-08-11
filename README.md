[![arXiv link](https://img.shields.io/badge/cs.CL%3A-arXiv%3A1606.08089-B31B1B.svg)](https://arxiv.org/abs/1606.08089)

# this-before-that
Causal precedence relations in the biomedical domain


# Getting the corpus (and other resources)
  The annotated corpus [can be found here](https://github.com/myedibleenso/this-before-that/blob/master/annotations.json?raw=true).  To download the word embeddings, however, you will need [`git-lfs`](https://git-lfs.github.com).

  Once you've installed `git-lfs`, simply run this command:

  ```
  git-lfs fetch
  ```

# Running the sieve-based architecture (*sans*-`lstm`)

## What you'll need...
  1. [Java 8](http://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html)
  2. [sbt](http://www.scala-sbt.org/release/tutorial/Setup.html)

# Rules used in the deterministic models

Three sets of `Odin`-style rules were used for the deterministic models:

1. [Inter-sentential patterns](https://github.com/clulab/reach/blob/master/src/main/resources/org/clulab/assembly/grammars/intersentential.yml)
2. [Intra-sentential patterns](https://github.com/clulab/reach/blob/master/src/main/resources/org/clulab/assembly/grammars/precedence.yml)
3. [Reichenbach rules for tense and aspect](https://github.com/clulab/reach/blob/master/src/main/resources/org/clulab/assembly/grammars/tense_aspect.yml)

# Running the `LSTM`

## Installation: Using `conda` and Python 3.X

1. Fork and clone [this repository](https://github.com/myedibleenso/this-before-that)

2. [Install `conda`](http://conda.pydata.org/miniconda.html)

2. Create a new `conda` environment using the [`environment.yml`](environment.yml) config:

```bash
conda env create -f environment.yml
```
The environment can be updated using the following command: 

```bash
conda env update -f environment.yml
```

3. Activate the environment:
```conda
source activate bionlp
```

4. Test the installation:
```python
python -c "import keras; print('Keras version: ', keras.__version__)"
```

### GPU training

If you're training on an Ubuntu system with a CUDA card, you can run `gpu_dependencies.sh` to set things up.

## Running the notebooks

```python
jupyter notebook
```
