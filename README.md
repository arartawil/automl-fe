# XAIevo: Explainable AI for Evolutionary Algorithms

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Making evolutionary and metaheuristic algorithms interpretable through comprehensive explainability tools.**

XAIevo is a general-purpose toolkit for adding explainability, interpretability, and transparency to evolutionary algorithms, metaheuristics, and nature-inspired optimization methods. It provides algorithm-agnostic interfaces for tracking, logging, analyzing, and visualizing the behavior of optimization processes.

## ✨ Key Features

- **🔍 Algorithm-Agnostic**: Works with any evolutionary or metaheuristic algorithm (DE, GA, PSO, CMA-ES, etc.)
- **📊 Run Tracking**: Comprehensive logging of iterations, fitness, diversity, and custom metrics
- **🎯 Attribution & Importance**: Compute feature importance and parameter sensitivity
- **📈 Visualization-Ready**: Export data in formats ready for plotting and analysis
- **🔌 Easy Integration**: Minimal code changes to add explainability to existing algorithms
- **💾 Persistence**: Save and reload optimization runs for post-hoc analysis
- **🎓 Research-Friendly**: Built for reproducibility and scientific investigation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/xaievo.git
cd xaievo

# Install in development mode
pip install -e .
```

### Minimal Example

```python
import sys
sys.path.insert(0, 'src')

from xaievo import RunTracker, Explainer
import random

# Create a tracker for your optimization run
tracker = RunTracker(
    run_id="sphere_optimization",
    config={"population_size": 50, "max_iterations": 100}
)

# Simulate a simple optimization loop (replace with your algorithm)
for iteration in range(100):
    best_fitness = 100.0 * (0.95 ** iteration)  # Simulated improvement
    diversity = random.uniform(0.1, 1.0)
    best_solution = [random.uniform(-5, 5) for _ in range(5)]
    
    # Log iteration data
    tracker.log_iteration(
        iteration=iteration,
        best_fitness=best_fitness,
        population_diversity=diversity,
        best_solution=best_solution
    )

# Finalize and analyze
tracker.finalize()
print(tracker.get_summary())

# Export for further analysis
tracker.export_json("optimization_run.json")

# Get explanations
explainer = Explainer(tracker)
print(explainer.analyze_convergence())
```

### Running the Demo

```bash
python main.py
```

## 📦 Project Structure

```
xaievo/
├── src/
│   └── xaievo/           # Main package
│       ├── __init__.py   # Package exports
│       ├── core.py       # Core explainability classes
│       └── cli.py        # Command-line interface
├── tests/                # Test suite
├── docs/                 # Documentation
├── examples/             # Example scripts
├── main.py               # Runnable demo
├── README.md             # This file
├── LICENSE               # GPL-3.0 license
├── CODE_OF_CONDUCT.md    # Contributor Covenant
├── CONTRIBUTING.md       # Contribution guidelines
└── requirements.txt      # Dependencies
```

## 🎯 Use Cases

### 1. Understanding Convergence Behavior
Track how your algorithm's fitness improves over time and identify convergence patterns.

### 2. Feature Importance in Optimization
Determine which decision variables have the most impact on solution quality.

### 3. Parameter Sensitivity Analysis
Understand how algorithm hyperparameters affect optimization performance.

### 4. Algorithm Comparison
Compare different optimization methods using standardized explainability metrics.

### 5. Reproducibility
Save complete run histories for reproducible research and debugging.

## 🔧 Advanced Usage

### Wrapping Custom Algorithms

```python
from xaievo import ExplainableOptimizer

def my_optimizer(objective_func, bounds, **kwargs):
    # Your optimization logic here
    pass

explainable_opt = ExplainableOptimizer(
    optimizer_func=my_optimizer,
    config={"method": "custom_ga", "pop_size": 100},
    enable_tracking=True
)

result = explainable_opt.optimize(
    objective_func=lambda x: sum(xi**2 for xi in x),
    bounds=[(-10, 10)] * 5
)

print(result["summary"])
print(result["explanations"])
```

### CLI Usage

```bash
# Run an optimization experiment
python -m xaievo.cli run --config config.json --output results/

# Analyze saved results
python -m xaievo.cli analyze results/run_20241226.json --report report.pdf
```

## 📊 Roadmap

### Version 0.1.0 (Current)
- ✅ Basic run tracking and logging
- ✅ Iteration snapshots with metadata
- ✅ JSON export functionality
- ✅ Simple convergence analysis

### Version 0.2.0 (Planned)
- 🔲 Feature importance computation methods
- 🔲 Population diversity metrics
- 🔲 Visualization module (convergence plots, heatmaps)
- 🔲 More analysis tools (stagnation detection, premature convergence)

### Version 0.3.0 (Planned)
- 🔲 Real-time monitoring dashboard
- 🔲 Integration with popular libraries (DEAP, PyGMO, Optuna)
- 🔲 Parameter sensitivity analysis tools
- 🔲 Comparative analysis across multiple runs

### Version 1.0.0 (Future)
- 🔲 Comprehensive documentation and tutorials
- 🔲 Paper submission and citation guidelines
- 🔲 Benchmarking suite
- 🔲 Plugin system for custom explainability methods

## 📚 Documentation

Full documentation is coming soon. For now, see:

- **Code**: Well-documented source code in [src/xaievo/](src/xaievo/)
- **Examples**: Example scripts in [examples/](examples/)
- **Tests**: Test cases in [tests/](tests/) show usage patterns

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs and requesting features
- Development setup and workflow
- Code style and testing requirements
- Pull request process

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

**TL;DR**: You can use, modify, and distribute this software, but any derivative work must also be open-source under GPL-3.0.

## 📖 How to Cite

If you use XAIevo in your research, please cite:

```bibtex
@software{xaievo2024,
  title = {XAIevo: Explainable AI for Evolutionary Algorithms},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/xaievo},
  version = {0.1.0},
  license = {GPL-3.0}
}
```

A research paper is in preparation. Citation will be updated once published.

## 🙏 Acknowledgments

- Inspired by explainability needs in evolutionary computation research
- Built with Python's scientific computing stack
- Thanks to the open-source evolutionary algorithms community

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/xaievo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/xaievo/discussions)
- **Email**: your.email@example.com

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub!

---

**Built for the evolutionary computation and XAI communities** 🧬🤖
