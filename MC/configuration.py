import random
import numpy as np
from copy import deepcopy
from collections import defaultdict

import torch

from pymatgen.core import Element, Structure

from utils import structureToGraph, getAtomicFeatures, setElementbyConcentration, preprocessDisorderSites

class Configuration():
    def __init__(self, structure, disorder_sites, disorder_species, starting_concentration, chemical_potential, atom_dict_type):
        disorder_sites = preprocessDisorderSites(structure, disorder_sites)
        self.disorder_species = disorder_species
        self.structure, self.disorder_element_table = self.setConcentration(
            structure, disorder_sites, disorder_species, starting_concentration
        )

        self.atom_dict = getAtomicFeatures(atom_dict_type)
        self.graph = structureToGraph(self.structure, atom_dict=self.atom_dict)

        if chemical_potential is None:
            chemical_potential = [0] * len(disorder_species)
        self.chemical_potential = {
            elem: mu for elem, mu in zip(disorder_species, chemical_potential)
        }

    def calculateChemicalPotential(self):
        energy = 0
        composition = self.getCurrentComposition()
        for elem in self.disorder_species:
            energy += composition.get(elem, 0) * self.chemical_potential[elem]
        return energy

    def setConcentration(self, structure, disorder_sites, disorder_species, concentration):
        species = structure.species.copy()
        shuffled_indices = setElementbyConcentration(disorder_sites, concentration)
        disorder_element_table = defaultdict(set)
        for i, site_idx in enumerate(disorder_sites):
            species[site_idx] = disorder_species[shuffled_indices[i]]
            disorder_element_table[disorder_species[shuffled_indices[i]]].add(site_idx)

        structure = Structure(
            lattice=structure.lattice, species=species, coords=structure.frac_coords, coords_are_cartesian=False
        )
        return structure, disorder_element_table

    def getCurrentStructure(self):
        return self.structure.copy()

    def getCurrentComposition(self):
        composition = self.structure.composition.to_data_dict['unit_cell_composition']
        for elem in self.disorder_species:
            if elem not in composition:
                composition[elem] = 0
        return composition

    def getCurrentConcentration(self):
        composition = self.getCurrentComposition()
        total_num_atms = sum(composition[elem] for elem in self.disorder_species)
        return {elem: (composition[elem] / total_num_atms if total_num_atms > 0 else 0) for elem in self.disorder_species}

    def getCurrentElementIndexTable(self):
        return deepcopy(self.disorder_element_table)

    def getConfigurationSummary(self):
        return {
            'composition': self.getCurrentComposition(),
            'concentration': self.getCurrentConcentration(),
            'element_to_atomic_idx_table': self.getCurrentElementIndexTable()
        }

    def copy(self):
        return deepcopy(self)

class FixedCompositionConfiguration(Configuration):
    def __init__(self, structure, disorder_sites, disorder_species,
                 starting_concentration, atom_dict_type='CGCNN', **kwargs):
        super().__init__(structure, disorder_sites, disorder_species, starting_concentration,
                         chemical_potential=None, atom_dict_type=atom_dict_type)

    def nextMove(self):
        new_configuration = self.copy()
        species = random.sample(new_configuration.disorder_species, 2)
        idx0 = random.choice(list(new_configuration.disorder_element_table[species[0]]))
        idx1 = random.choice(list(new_configuration.disorder_element_table[species[1]]))

        new_configuration.structure[idx0] = species[1]
        new_configuration.structure[idx1] = species[0]
        new_configuration.graph.x[[idx0, idx1]] = new_configuration.graph.x[[idx1, idx0]]
        ## graph.x and graph.z point to the same object, so no need to change both
        # new_configuration.graph.z[[idx0, idx1]] = new_configuration.graph.z[[idx1, idx0]]

        new_configuration.disorder_element_table[species[0]].remove(idx0)
        new_configuration.disorder_element_table[species[0]].add(idx1)
        new_configuration.disorder_element_table[species[1]].remove(idx1)
        new_configuration.disorder_element_table[species[1]].add(idx0)
        return new_configuration

class VariableCompositionConfiguration(Configuration):
    def __init__(self, structure, disorder_sites, disorder_species,
                 starting_concentration, chemical_potential, concentration_constraints={}, atom_dict_type='CGCNN'):
        super().__init__(structure, disorder_sites, disorder_species, starting_concentration,
                         chemical_potential=chemical_potential, atom_dict_type=atom_dict_type)

        self.concentration_constraints = concentration_constraints

    def satisfyConcentrationConstraints(self, concentration):
        if not self.concentration_constraints:
            return True
        for elem, c in concentration.items():
            if elem in self.concentration_constraints:
                if c > self.concentration_constraints[elem][1] or c < self.concentration_constraints[elem][0]:
                    return False
        return True

    def nextMove(self, max_num_trials=50):
        num_trial = 1
        while num_trial < max_num_trials:
            new_configuration = self.copy()
            species = random.sample(new_configuration.disorder_species, 2)
            if not new_configuration.disorder_element_table[species[0]]:
                species = species[::-1]

            idx0 = random.choice(list(new_configuration.disorder_element_table[species[0]]))

            new_configuration.structure[idx0] = species[1]
            new_configuration.graph.x[idx0] = torch.tensor(self.atom_dict[Element(species[1]).Z])
            ## graph.x and graph.z point to the same object, so no need to change both
            # new_configuration.graph.z[idx0] = torch.tensor(self.atom_dict[Element(species[1]).Z])

            new_configuration.disorder_element_table[species[0]].remove(idx0)
            new_configuration.disorder_element_table[species[1]].add(idx0)

            new_concentration = new_configuration.getCurrentConcentration()
            if self.satisfyConcentrationConstraints(new_concentration):
                return new_configuration

            num_trial += 1

        raise ValueError(f'Under the concentration {self.getCurrentConcentration()}, '
                         f'the next move cannot be proposed after {max_num_trials} steps '
                         f'under the constrains {self.concentration_constraints}')

if __name__ == '__main__':
    structure = Structure.from_file('supercell.vasp')
    conf = FixedCompositionConfiguration(structure, disorder_sites='O', disorder_species=['O', 'F'], starting_concentration=[0.9, 0.1],
                                            concentration_constraints={'O': [0.5, 1],})
    new_conf = conf.nextMove(max_num_trials=2)