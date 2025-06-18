import os
import pickle as pkl
import numpy as np
from tqdm import tqdm

import torch

from pymatgen.core import Structure

from models import PeriodicNetwork

from configuration import VariableCompositionConfiguration, FixedCompositionConfiguration
from postprocessing import PostProcessing_Metropolis

from utils import denormalize, getLogger

from settings import kB

class Metropolis():
    def __init__(self, model, dataset_metadata,
                 structure, disorder_sites, disorder_species, atom_dict_type,
                 starting_concentration, fix_concentration, concentration_constraints=None, chemical_potential=None,
                 Tmax=1000, Tmin=10, dT=10, num_steps_per_T=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.out_dim = [1, 251, 91]

        self.dataset_metadata = dataset_metadata

        if fix_concentration:
            self.configuration = FixedCompositionConfiguration(
                structure=structure, disorder_sites=disorder_sites, disorder_species=disorder_species,
                 starting_concentration=starting_concentration, atom_dict_type=atom_dict_type
            )
        else:
            if not chemical_potential:
                raise ValueError(f'You have to specify the chemical potential for the disordering sites when doing '
                                 f'grand canonical Monte Carlo simulation.')
            self.configuration = VariableCompositionConfiguration(
                structure=structure, disorder_sites=disorder_sites, disorder_species=disorder_species,
                 starting_concentration=starting_concentration, concentration_constraints=concentration_constraints,
                 chemical_potential=chemical_potential, atom_dict_type=atom_dict_type
            )

        self.configuration_info = {
            'structure': structure,
            'disorder_sites': disorder_sites,
            'disorder_species': disorder_species,
            'starting_concentration': starting_concentration,
            'fix_concentration': fix_concentration,
            'concentration_constraints': concentration_constraints
        }

        self.num_steps_per_T = num_steps_per_T
        self.Tgrid = np.arange(Tmin, Tmax + dT, dT)[::-1]

        species_str = "[" + ", ".join(disorder_species) + "]"
        concentration_str = "[" + ", ".join(f"{c}" for c in starting_concentration) + "]"
        self.name = f'Metropolis_{species_str}_{concentration_str}'
        self.logger = getLogger(f'logfiles/{self.name}')

    def evaluateConfiguration(self, conf):
        def eval(model, conf):
            model.eval()
            with torch.no_grad():
                graph =  conf.graph.clone().to(self.device)
                output = model(graph).detach().cpu().numpy()
            return np.squeeze(output)

        output = eval(self.model, conf)

        energy = output[: self.out_dim[0]]
        optcond = output[self.out_dim[0] : self.out_dim[0] + self.out_dim[1]]
        elecond = output[self.out_dim[0] + self.out_dim[1] : ]
        print(denormalize(energy, self.dataset_metadata['energy']['stats']), - conf.calculateChemicalPotential())
        print(conf.getCurrentComposition())
        return {
            'energy': denormalize(energy, self.dataset_metadata['energy']['stats']) - conf.calculateChemicalPotential(),
            # 'energy': denormalize(output[0], self.dataset_metadata['energy']['stats']),
            'optcond': denormalize(optcond, self.dataset_metadata['optcond']['stats']),
            'elecond': denormalize(elecond, self.dataset_metadata['elecond']['stats'])
        }

    def accept(self, E_old, E_new, T):
        dE = E_new - E_old
        if dE < 0:
            return True
        return np.random.uniform() <= np.exp(-dE / (kB * T))

    def run(self):
        if not os.path.exists(f'./save/MC/{self.name}/results'):
            os.makedirs(f'./save/MC/{self.name}/results')

        MC_summary = {
            'Tgrid': self.Tgrid,
            'num_steps_per_T': self.num_steps_per_T,
            'dataset_metadata': self.dataset_metadata,
            'configuration_info': self.configuration_info
        }
        with open(f'./save/MC/{self.name}/summary.pkl', 'wb') as f:
            pkl.dump(MC_summary, f)

        self.logger.info(f'Staring MC simulation with Metropolis sampling.\n '
                         f'The temperature range is {self.Tgrid[0]} ~ {self.Tgrid[-1]} K, '
                         f'with {self.num_steps_per_T} steps for each temperature.')

        configuration_cur = self.configuration.copy()
        output_cur = self.evaluateConfiguration(configuration_cur)
        for T in self.Tgrid:
            self.logger.info(f'Current temperature is {T} K.')
            summary = []
            # for i in tqdm(range(self.num_steps_per_T), total=self.num_steps_per_T):
            for i in range(self.num_steps_per_T):
                configuration_next = configuration_cur.nextMove()
                output_next = self.evaluateConfiguration(configuration_next)

                if self.accept(output_cur['energy'], output_next['energy'], T):
                    self.logger.info(f"Step {i}: E_old = {output_cur['energy']} eV, "
                                     f"E_new = {output_next['energy']}.eV, accepted")
                    summary.append({
                        'step': i, 'T': T, 'output': output_next,
                        'configuration_summary': configuration_next.getConfigurationSummary()
                    })
                    configuration_cur = configuration_next
                    output_cur = output_next
                else:
                    self.logger.info(f"Step {i}: E_old = {output_cur['energy']} eV, "
                                     f"E_new = {output_next['energy']}.eV, rejected")
                    summary.append({
                        'step': i, 'T': T, 'output': output_cur,
                        'configuration_summary': configuration_cur.getConfigurationSummary()
                    })
            with open(f'./save/MC/{self.name}/results/{T}K.pkl', 'wb') as f:
                pkl.dump(summary, f)

if __name__ == '__main__':
    structure = Structure.from_file('supercell.vasp')

    with open('pretrained_model/dataset_metadata/mean_all.pkl', 'rb') as f:
        dataset_metadata = pkl.load(f)
    dataset_metadata['optcond']['stats']['mean_val'] = np.squeeze(dataset_metadata['optcond']['stats']['mean_val'], axis=0)
    dataset_metadata['elecond']['stats']['mean_val'] = np.squeeze(dataset_metadata['elecond']['stats']['mean_val'], axis=0)

    model = PeriodicNetwork(
        in_dim=92,
        em_dim=64,
        irreps_in=str(64) + "x0e",
        irreps_out=str(1) + "x0e+" + str(251) + "x0e+" + str(91) + "x0e",
        irreps_node_attr=str(64) + "x0e",
        layers=2,
        mul=32,
        lmax=2,
        max_radius=6,
        num_neighbors=63,
        reduce_output=True,
    )
    modelname = 'E3NN_2_64_32_2_0.005_0.05_0_4_model'  # replace with trained modelname
    model.load_state_dict(torch.load(f'pretrained_model/pretrained_GNN_model/{modelname}.torch', map_location='cpu'))

    mc = Metropolis(
        model=model,
        dataset_metadata=dataset_metadata,
        structure=structure,
        disorder_sites=['O'],
        disorder_species=['O', 'F'],
        atom_dict_type='CGCNN',
        starting_concentration=[0.80, 0.20],
        fix_concentration=False,
        concentration_constraints={'O': [0.5, 1], 'F': [0, 0.5]},
        chemical_potential=[-4.614270, -1.463544],
        Tmax=1000,
        Tmin=100,
        dT=10,
        num_steps_per_T=10,
    )
    results = mc.run()

    pp = PostProcessing_Metropolis(MC_name=mc.name)
    pp.run(chosen_ratio=0.75, average_method='average', savefig_dir_suffix='average')