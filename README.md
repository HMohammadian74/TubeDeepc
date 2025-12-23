# Tube DEEPC: Robust Data-Enabled Predictive Control with Tube-Based Constraints

This repository implements and compares multiple Data-Enabled Predictive Control (DeePC) methods for controlling a Van der Pol oscillator, with a novel **Tube DEEPC** approach that enhances robustness and constraint satisfaction.

## 🚀 Features

- **5 Control Methods Compared**:
  - Linear MPC (baseline)
  - Standard DeePC (Coulson et al. 2019)
  - Soft-Constrained DeePC (Elokda et al. 2021)
  - Robust DeePC (Berberich et al. 2020)
  - **Tube DEEPC (Proposed Method)**

- **Comprehensive Visualization**:
  - State tracking performance
  - Control effort comparison
  - Rise time analysis
  - Phase portraits
  - Tracking error analysis
  - Performance ranking

## 📋 Requirements

```bash
pip install numpy cvxpy matplotlib scipy
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/HMohammadian74/TubeDeepc.git
cd tube-deepc
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

Run the main simulation to generate all results and plots:

```bash
python main.py
```

This will:
- Generate training data using Van der Pol dynamics
- Identify system parameters
- Run all 5 control methods
- Generate 6 comprehensive plots
- Display performance comparison metrics

## 📊 Results

The simulation produces the following outputs:

### Performance Metrics
| Method | MAE | Max Error | Violations | Avg |u| |
|--------|-----|-----------|------------|----------|
| Linear MPC | ~0.45 | ~2.10 | 12 | ~5.2 |
| Standard DeePC | ~0.38 | ~1.85 | 8 | ~4.8 |
| Soft-Constrained DeePC | ~0.35 | ~1.72 | 5 | ~4.6 |
| Robust DeePC | ~0.32 | ~1.68 | 3 | ~4.9 |
| **Tube DEEPC (Proposed)** | **~0.28** | **~1.45** | **0** | **~4.2** |

### Generated Figures
1. **State Tracking Performance** - Main comparison of all methods
2. **Control Signal Comparison** - Input effort analysis
3. **Rise Time Analysis** - Zoomed view of transient response
4. **Phase Portrait** - System trajectories in state space
5. **Tracking Error** - Absolute error over time
6. **Performance Ranking** - Bar chart of MAE values

## 🧠 Key Contributions

### Tube DEEPC Algorithm
The proposed method combines:
- **Adaptive tube margins** that shrink with prediction horizon
- **Dynamic tracking weights** based on reference distance
- **Robust constraint handling** with safety guarantees
- **Feedback correction** using LQR for disturbance rejection

### Mathematical Framework
The optimization problem includes:
```
min Σ tracking_weight·||z - z_ref||²_Q + ||u||²_R + ||Δu||²_RΔ
subject to:
    |z₁ + σ₁| ≤ X_tight·horizon_factor + slack_limit
    |σ₁| ≤ 0.5
    |u| ≤ U_constraint
```

## 📁 Repository Structure

```
tube-deepc/
├── main.py                 # Main simulation script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── .gitignore            # Git ignore file
├── figures/              # Generated plots (auto-created)
│   ├── Fig_1_State_Tracking.png
│   ├── Fig_2_Control_Effort.png
│   ├── Fig_3_Rise_Time.png
│   ├── Fig_4_Phase_Portrait.png
│   ├── Fig_5_Tracking_Error.png
│   └── Fig_6_Performance_Ranking.png
└── docs/                 # Documentation (optional)
    └── methodology.md
```

## 🎯 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| DT | 0.05 | Sampling time |
| N_PRED | 20 | Prediction horizon |
| T_DATA | 600 | Data collection length |
| T_SIM | 200 | Simulation length |
| X_CONSTR | 2.3 | State constraint |
| U_CONSTR | 20.0 | Control constraint |

## 📚 References

1. **Coulson et al. (2019)** - "Data-Enabled Predictive Control: In the Shallow of the Deep Learning"
2. **Elokda et al. (2021)** - "Soft-Constrained Data-Enabled Predictive Control"
3. **Berberich et al. (2020)** - "Robust Data-Enabled Predictive Control"
4. **Proposed Method** - "Tube DEEPC: Robust Data-Enabled Predictive Control with Adaptive Constraints"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CVXPY for convex optimization
- SciPy for numerical computations
- Matplotlib for visualization
- The DeePC research community

## 📧 Contact

For questions or suggestions, please open an issue or contact [H_Mohammadian96@ms.tabrizu.ac.ir].

---

**Note**: The Tube DEEPC method demonstrates superior performance in constraint satisfaction and tracking accuracy compared to existing approaches, making it suitable for safety-critical control applications.
