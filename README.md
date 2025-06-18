# A Machine Learning Framework for Modeling Ensemble Properties of Atomically Disordered Materials

## Description

This work proposes a general statistical framework involving graph neural networks and Monte-Carlo simulations to predict the thermodynamic properties and ensemble-averged functional properties of disordered materials.

## Dependencies

All required packages are listed in `requirements.txt`. Some key dependencies include:

- `python==3.11`
- `torch==2.3.1`
- `torch_geometric==2.5.3`
- `e3nn==0.5.5`

You can install them with:

```bash
pip install -r requirements.txt
```

## Usage

For graph neural network training and testing, navigate to the `GNN/` directory, and run the script:
```bash
python GNN.py
```

For Monte-Carlo simulation, nativate to the `MC/` directory, and run the script:
```bash
python Metropolis.py
```

## Citation

If you find this work useful, please consider cite the following reference:


