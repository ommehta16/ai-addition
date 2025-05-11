# ✨AI✨ Addition

## "21st century problems require 21st century solutions"

A Python-based CPU-trained neural network for adding numbers.

This solves a problem that everyone has, but *no one* has invented a solution for yet: how to make a computer add numbers together. It's a **massive** revolution in computing.

### Prerequisites

Everything except for `numpy` should come with python by default:

- `pip3 install numpy`

Other than those, you only need `os`, `sys`, `json`, and `datetime`.

### Running

- Simply run `main.py` with your installed python version, with the following args:
  - `-run [filename]` to run the best-performing (first) set of weights from the training data file
  - `-train` to re-train the model

### Tuning

- I haven't made any *actual* args for this (you probably don't need/want them anyways)
- BUT if you do want them, the number of agents per generation (`N`), number of testcases(`T`), number of generations(`G`), network structure, and survival ratio are all constants at the top of main.py -- feel free to edit them :)

### WHY?

- idk it's funny
- "education"
- lowk just wanted to do this to figure out who AI is 🤷‍♂️
