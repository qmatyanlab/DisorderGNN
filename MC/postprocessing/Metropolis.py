import os
import pickle as pkl
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from settings import kB

class PostProcessing_Metropolis():
    def __init__(self, MC_name):
        self.name = MC_name

        with open(f'save/MC/{self.name}/summary.pkl', 'rb') as f:
            MC = pkl.load(f)

        self.Tgrid = MC['Tgrid']
        self.num_steps_per_T = MC['num_steps_per_T']
        self.dataset_metadata = MC['dataset_metadata']
        self.configuration_info = MC['configuration_info']
        self.disorder_species = MC['configuration_info']['disorder_species']
        self.starting_concentration = MC['configuration_info']['starting_concentration']

    def averageConductivity(self, conductivity, method):
        if method == 'average':
            if conductivity.ndim == 1:
                return np.mean(conductivity)
            elif conductivity.ndim == 2:
                return np.mean(conductivity, axis=0)
        elif method == 'inverse_average':
            if conductivity.ndim == 1:
                return 1 / (np.mean(1 / conductivity))
            elif conductivity.ndim == 2:
                return 1 / (np.mean(1 / conductivity, axis=0))

    def loadResultsAtIndividualTemp(self, T):
        with open(f'save/MC/{self.name}/results/{T}K.pkl', 'rb') as f:
            summary = pkl.load(f)
        return summary

    def analyzeComposition(self, chosen_ratio, savefig_dir):
        compositions = []
        for T in self.Tgrid:
            results = self.loadResultsAtIndividualTemp(T)
            compositions.append([summary['configuration_summary']['composition'] for summary in results])

        start_idx = int(self.num_steps_per_T * (1 - chosen_ratio))
        ave_compositions = defaultdict(list)
        for i, composition in enumerate(compositions):
            composition_chosen = composition[start_idx : ]
            for elem in self.disorder_species:
                comp_array = np.array([comp[elem] for comp in composition_chosen])
                ave_compositions[elem].append(np.mean(comp_array))

        fig, ax = plt.subplots()
        colormap = plt.get_cmap("viridis", len(ave_compositions.keys()))
        for i, elem in enumerate(self.disorder_species):
            comp = ave_compositions[elem]
            ax.plot(self.Tgrid, comp, color=colormap(i), label=elem)
        plt.legend()
        plt.savefig(f'{savefig_dir}/compositions_ave.pdf')

        with open(f'{savefig_dir}/compositions_ave.dat', 'w') as f:
            keys = list(ave_compositions.keys())
            f.write("# Tgrid " + " ".join(keys) + "\n")
            for i in range(len(self.Tgrid)):
                row = [f"{self.Tgrid[i]}"]
                for elem in keys:
                    row.append(f"{ave_compositions[elem][i]}")
                f.write(" ".join(row) + "\n")

    def analyzeEnergyAndHeatCapacity(self, chosen_ratio, savefig_dir):
        T_ticks = [1000, 800, 600, 400, 200]

        energies = []
        for T in self.Tgrid:
            results = self.loadResultsAtIndividualTemp(T)
            energies.append(np.array([summary['output']['energy'] for summary in results]))
        energies_flat = np.concatenate(energies)
        min_energy = np.min(energies_flat)

        fig, ax = plt.subplots()
        ax.plot(range(len(energies_flat)), energies_flat, color='black')
        xtick_positions = []
        for T in T_ticks:
            T_idx = np.where(self.Tgrid == T)[0][0]
            pos = T_idx * self.num_steps_per_T
            xtick_positions.append(pos)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(T_ticks)
        plt.savefig(f'{savefig_dir}/energies_full.pdf')

        ## average energies, heat capacity
        start_idx = int(self.num_steps_per_T * (1 - chosen_ratio))
        ave_energies = np.array([
            np.mean(energy[start_idx : ] - min_energy)
            for energy in energies
        ])
        ave_energies_squared = np.array([
            np.mean((energy[start_idx : ] - min_energy)**2)
            for energy in energies
        ])
        heat_capacity = (ave_energies_squared - ave_energies ** 2) / (kB * self.Tgrid ** 2)

        with open(f'{savefig_dir}/energies_ave.dat', 'w') as f:
            f.write('# T\taverage energies\n')
            for T, E in zip(self.Tgrid, ave_energies):
                f.write(f'{T}\t{E}\n')

        fig, ax = plt.subplots()
        ax.plot(self.Tgrid, ave_energies, color='black')
        ax.set_xticks(T_ticks)
        plt.gca().invert_xaxis()
        plt.savefig(f'{savefig_dir}/energies_ave.pdf')

        with open(f'{savefig_dir}/heat_capacity.dat', 'w') as f:
            f.write('# T\theat capacity\n')
            for T, Cv in zip(self.Tgrid, heat_capacity):
                f.write(f'{T}\t{Cv}\n')

        fig, ax = plt.subplots()
        ax.plot(self.Tgrid, heat_capacity, color='black')
        ax.set_xticks(T_ticks)
        plt.savefig(f'{savefig_dir}/heat_capacity.pdf')

    def analyzeOpticalConductivity(self, average_method, chosen_ratio, savefig_dir):
        optconds = []
        for T in self.Tgrid:
            results = self.loadResultsAtIndividualTemp(T)
            optconds.append(np.array([summary['output']['optcond'] for summary in results]))

        start_idx = int(self.num_steps_per_T * (1 - chosen_ratio))
        ave_optconds = np.array([
            self.averageConductivity(optcond[start_idx :, :], average_method)
            for optcond in optconds
        ])

        xgrid = self.dataset_metadata['optcond']['xgrid']
        indices = [i for i, temp in enumerate(self.Tgrid) if temp % 100 == 0]
        selected_temp = [self.Tgrid[i] for i in indices]
        selected_data = ave_optconds[indices, :].T

        with open(f'{savefig_dir}/optcond_ave.dat', 'w') as f:
            f.write("# hw\t" + "\t".join(f'{t}K' for t in selected_temp) + "\n")
            for omega, row in zip(xgrid, selected_data):
                f.write(f'{omega:.2f}\t' + "\t".join(f"{val:.6f}" for val in row) + "\n")

        fig, ax = plt.subplots(figsize=(10, 6))

        colormap = plt.get_cmap('afmhot')
        norm = mcolors.Normalize(vmin=min(self.Tgrid[indices]), vmax=max(self.Tgrid[indices]))
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        for i, idx in enumerate(indices):
            row = ave_optconds[idx, :]
            color = colormap(i / len(indices))
            ax.plot(xgrid, row, color=color)
        ax.set_title('average optical conductivity at each temperature')
        ax.set_xlabel('photon energy')
        ax.set_ylabel('average optical conductivity')

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Temperature (K)', rotation=270, labelpad=15)
        plt.savefig(f'{savefig_dir}/optcond_ave.pdf')

    def analyzeElectricalConductivity(self, average_method, chosen_ratio, savefig_dir):
        eleconds = []
        for T in self.Tgrid:
            results = self.loadResultsAtIndividualTemp(T)
            eleconds.append(np.array([summary['output']['elecond'] for summary in results]))

        start_idx = int(self.num_steps_per_T * (1 - chosen_ratio))
        ave_eleconds = []
        for i, elecond in enumerate(eleconds):
            ave_elecond = self.averageConductivity(elecond[start_idx :, :], average_method)

            original_xgrid = self.dataset_metadata['elecond']['xgrid']
            new_xgrid = self.Tgrid
            ave_elecond = interp1d(original_xgrid, ave_elecond, kind='quadratic', fill_value="extrapolate")(new_xgrid)
            ave_eleconds.append(ave_elecond[i])
        ave_eleconds = np.array(ave_eleconds)

        with open(f'{savefig_dir}/elecond_ave.dat', 'w') as f:
            f.write('# T\taverage electrical conductivity\n')
            for T, cond in zip(self.Tgrid, ave_eleconds):
                f.write(f'{T}\t{cond}\n')

        T_ticks = [1000, 800, 600, 400, 200]
        fig, ax = plt.subplots()
        ax.plot(self.Tgrid, ave_eleconds, color='black')
        ax.set_xticks(T_ticks)
        ax.set_xlabel('temperature')
        ax.set_ylabel('average electrical conductivity')
        plt.savefig(f'{savefig_dir}/elecond_ave.pdf')

    def plotExampleConductivity(self, T, savefig_dir, num_conductivity=10):
        results = self.loadResultsAtIndividualTemp(T)
        energies = np.array([summary['output']['energy'] for summary in results])
        optconds = np.array([summary['output']['optcond'] for summary in results])
        omega = self.dataset_metadata['optcond']['xgrid']
        eleconds = np.array([summary['output']['elecond'] for summary in results])
        temperature = self.dataset_metadata['elecond']['xgrid']

        selected_indices = np.random.choice(optconds.shape[0], size=num_conductivity, replace=False)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].plot(range(len(energies)), energies, color='black')
        axes[0].set_title(f'energies at temperature {T}K')
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('energy')

        color_map = plt.get_cmap("viridis", num_conductivity)
        for i, idx in enumerate(selected_indices):
            color = color_map(i)
            axes[1].plot(omega, optconds[idx], color=color, alpha=0.6)
            axes[2].plot(temperature, eleconds[idx], color=color, alpha=0.6)
        axes[1].set_title(f"optcond at temperature {T}K")
        axes[1].set_xlabel('photon energy')
        axes[2].set_ylabel('conductivity')
        axes[2].set_title(f"elecond at temperature {T}K")
        axes[2].set_xlabel('temperature')
        axes[2].set_ylabel('conductivity')

        plt.tight_layout(rect=[0, 0, 1, 0.92])  # adjust the top padding from 0.95 to 0.92
        plt.subplots_adjust(top=0.85)  # tighter spacing between suptitle and subplots
        plt.savefig(f'{savefig_dir}/{T}K.pdf')

    def run(self, chosen_ratio=0.75, average_method='average', savefig_dir_suffix=None):
        if savefig_dir_suffix is None:
            savefig_dir = f'save/figs/{self.name}'
        else:
            savefig_dir = f'save/figs/{self.name}/{savefig_dir_suffix}'

        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)

        for T in [1000, 100, 750, 500, 250]:
            self.plotExampleConductivity(T, savefig_dir=savefig_dir)

        self.analyzeEnergyAndHeatCapacity(chosen_ratio=chosen_ratio, savefig_dir=savefig_dir)
        self.analyzeOpticalConductivity(average_method=average_method, chosen_ratio=chosen_ratio, savefig_dir=savefig_dir)
        self.analyzeElectricalConductivity(average_method=average_method, chosen_ratio=chosen_ratio, savefig_dir=savefig_dir)

        if not self.configuration_info['fix_concentration']:
            self.analyzeComposition(chosen_ratio=chosen_ratio, savefig_dir=savefig_dir)