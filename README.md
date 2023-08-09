# Reduction Method to resolve Contextual Bandit Problems

This repository contains implementations and comparisons of two contextual bandit algorithms (convenetional method v.s. reduction method):

1. **LinUCB**: A conventional contextual bandit algorithm that uses the linear upper confidence bound (LinUCB) strategy.
2. **Reduction Approach**: An approach that applies a reduction technique to fix the action set from the contextual distribution.

## Getting Started

To run the comparison and analyze the performance of the algorithms, follow the instructions below.

### Prerequisites

Make sure you have the following installed on your system:

- Python (>= 3.6)
- Required Python packages (specified in `requirements.txt`)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/contextual-bandit-comparison.git
   cd contextual-bandit-comparison
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments

To run the experiments and compare the performance of the two algorithms:

1. Run the comparison script:

   ```bash
   python comparison.py
   ```

The script will execute simulations of both algorithms and compare their performance in terms of regret.

## Results

The `comparison.py` script executes simulations for both the LinUCB algorithm and the reduction approach. It collects data on the regret of each algorithm over time and presents the results in a plot.

## Conclusion

The comparison results highlight the performance of the two algorithms in terms of regret. This analysis provides insights into the effectiveness of the reduction approach compared to the conventional LinUCB algorithm.

Feel free to explore the code in this repository to understand the implementations of both algorithms and customize the simulations as needed.
