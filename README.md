# this-before-that
Causal precedence relations in the biomedical domain


# Getting the corpus (and other resources)
  To retrieve the annotated corpus, you will need [`git-lfs`](https://git-lfs.github.com).

  Once you've installed `git-lfs`, simply run this command:

  ```
  git-lfs fetch
  ```

# Running the sieve-based architecture (*sans*-`lstm`)

## What you'll need...
  1. [Java 8](http://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html)
  2. [sbt](http://www.scala-sbt.org/release/tutorial/Setup.html)

# Running the `LSTM`

## Installation: Using `conda` and Python 3.X

1. Fork and clone [this repository](https://github.com/myedibleenso/this-before-that)

2. [Install `conda`](http://conda.pydata.org/miniconda.html)

2. Create a new `conda` environment using the [`environment.yml`](environment.yml) config:

```bash
conda env create -f environment.yml
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
