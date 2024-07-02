import os, shutil, gzip
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from utils import remove_from_sdf, get_sdf_scores


class SminaDocking:
    def __init__(self, receptor_file, ligand_file, output_dir, known_binder=None, bind_residues=None, seed=0, num_nodes=50):
        
        if not receptor_file.endswith('pdbqt'):
            raise Exception('The receptor protein should be in PDBQT format.')
        if not ligand_file.endswith('sdf'):
            raise Exception('The small molecules (ligands) should be in SDF format.')
        if known_binder and bind_residues:
            raise Exception('Either known binder or binding residues should be given (NOT both of them).')
        
        self.receptor_file = receptor_file
        self.ligand_file = ligand_file
        self.output_dir = output_dir
        self.docking_file = os.path.join(output_dir,'smina_results.sdf')
        self.log_file = os.path.join(output_dir,'smina.log')
        self.known_binder = known_binder
        self.bind_residues = bind_residues
        self.seed = seed
        self.num_nodes = num_nodes
        
        self.mol_parser = PDBParser(PERMISSIVE=1)
        
    
    def run(self):        
        if self.known_binder:
            search_space = '--autobox_ligand ' + self.known_binder
            self._run_smina(search_space)
            # postprocess
            binder = self.mol_parser.get_structure('binder',self.known_binder)
            binder_coords = np.array([atom.get_coord() for atom in binder.get_atoms()])
            remove_from_sdf(binder_coords, self.docking_file, 3)
        
        elif self.bind_residues:
            res1, res2 = [int(res) for res in self.bind_residues.split('-')]
            receptor = self.mol_parser.get_structure('receptor',self.receptor_file)
            bsite_coords = []
            for res in receptor.get_residues():
                if res1 <= res.get_id()[1] <= res2:
                    bsite_coords.extend([atom.get_coord() for atom in res])
            bsite_coords = np.array(bsite_coords)
            search_space = self._get_search_bbox(bsite_coords)
            self._run_smina(search_space)
            # postprocess
            remove_from_sdf(bsite_coords, self.docking_file, 3)
        
        else:
            receptor = self.mol_parser.get_structure('receptor',self.receptor_file)
            mol_coords = np.array([atom.get_coord() for atom in receptor.get_atoms()])
            search_space = self._get_search_bbox(mol_coords)
            self._run_smina(search_space)
            # no postprocess
    
    def get_affinities(self):
        return get_sdf_scores(self.docking_file)
    
    def compress_output(self):
        with open(self.docking_file, 'rb') as f_in, gzip.open(self.docking_file+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(self.docking_file)
    
    def _get_search_bbox(self, coords):
        center = np.average(coords,axis=0)
        size = np.max(coords,axis=0) - np.min(coords,axis=0)
        return '--center_x {} --center_y {} --center_z {} --size_x {} --size_y {} --size_z {}'.format(
            center[0], center[1], center[2], size[0], size[1], size[2],
            )

    def _run_smina(self, search_space):
        smina_exe = 'smina/smina.static'
        receptor = '-r ' + self.receptor_file
        ligand = '-l ' + self.ligand_file
        output = '-o ' + self.docking_file
        log = '--log ' + self.log_file
        params = '--seed ' + str(self.seed) + ' --num_modes ' + str(self.num_nodes) + ' -q'

        os.system(' '.join([smina_exe, receptor, ligand, search_space, output, log, params]))


    