from scoring import cnn_rescoring, joined_score
import os, argparse, time, shutil, gzip, numpy as np
from Bio.PDB.PDBParser import PDBParser
from utils import remove_from_sdf, txt_to_npy
from pybel import readfile


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--receptor', '-r', required=True, help='protein receptor (.pdbqt)')
    parser.add_argument('--receptor_H', '-rH', required=True, help='protonated protein receptor (with hydrogens added) (.pdbqt)')
    parser.add_argument('--ligand', '-l', required=True, help='file with screening compounds (.sdf)')
    parser.add_argument('--known_binder', '-kb', help='known binder molecule (to determine screening site)')  
    parser.add_argument('--residues', '-res', help='target residues (e.g. in this format 85-100)')
    parser.add_argument('--output', '-o', required=True, help='output directory')

    return parser.parse_args()


args = parse_args()

# Check if inputs are valid

if not args.receptor.endswith('pdbqt') or not args.receptor_H.endswith('pdbqt'):
    raise Exception('The receptor protein should be in PDBQT format.')
if not args.ligand.endswith('sdf'):
    raise Exception('The small molecules (ligands) should be in SDF format.')
if args.known_binder and args.residues:
    raise Exception('Either known binder or binding residues should be given (NOT both of them).')

mol_parser = PDBParser(PERMISSIVE=1)

if not os.path.exists(args.output):
    os.makedirs(args.output)
t1 = time.time()

# Docking (smina)

docking_file = os.path.join(args.output,'smina_results.sdf')

if args.known_binder is not None:
    os.system('smina/smina.static -r '+args.receptor_H+' -l '+args.ligand+' --autobox_ligand '+args.known_binder+' -o '+docking_file+ 
          ' --log '+os.path.join(args.output,'smina.log')+ ' --seed 0 --num_modes 50 -q')
    t2 = time.time()
    mol = next(readfile(args.known_binder.split('.')[-1],args.known_binder))
    mol_coords = np.array([atom.coords for atom in mol.atoms])
    docking_file = remove_from_sdf(mol_coords,docking_file,3)
    t3 = time.time()
elif args.residues is not None:
    res1,res2 = args.residues.split('-')
    mol = mol_parser.get_structure('mol',args.receptor)
    mol_coords = []
    for res in mol.get_residues():
        if int(res1) <= res.get_id()[1] <= int(res2):
            mol_coords += [atom.get_coord() for atom in res]
    mol_center = np.average(mol_coords,axis=0)
    mol_coords = np.array(mol_coords)
    mol_size = np.max(mol_coords,axis=0) - np.min(mol_coords,axis=0)
    os.system('smina/smina.static -r '+args.receptor_H+' -l '+args.ligand+' --center_x '+str(mol_center[0])+' --center_y '+str(mol_center[1])+' --center_z '+str(mol_center[2])+
              ' --size_x '+str(mol_size[0])+' --size_y '+str(mol_size[1])+' --size_z '+str(mol_size[2])+' -o '+docking_file+
              ' --log '+os.path.join(args.output,'smina.log')+ ' --seed 0 --num_modes 50 -q')  
    t2 = time.time()
    docking_file = remove_from_sdf(mol_coords,docking_file,3)
    t3 = time.time()
else:
    mol = readfile(args.receptor.split('.')[-1],args.receptor).next()
    mol_coords = np.array([atom.coords for atom in mol.atoms])
    mol_center = np.average(mol_coords,axis=0)
    mol_size = np.max(mol_coords,axis=0) - np.min(mol_coords,axis=0)
    os.system('smina/smina.static -r '+args.receptor_H+' -l '+args.ligand+' --center_x '+str(mol_center[0])+' --center_y '+str(mol_center[1])+' --center_z '+str(mol_center[2])+
              ' --size_x '+str(mol_size[0])+' --size_y '+str(mol_size[1])+' --size_z '+str(mol_size[2])+' -o '+docking_file+
              ' --log '+os.path.join(args.output,'smina.log')+ ' --seed 0 --num_modes 50 -q')
    t2 = time.time()

# Get gnina features
    
receptor_gnina = args.receptor.rsplit('.',1)[0] + '_gninatypes.npy'

if not os.path.exists(receptor_gnina):
    receptor_path = args.receptor.rsplit('/',1)[0]
    os.system('gninatyper/build/gninatyper.out '+args.receptor+' '+receptor_path+' mol 1')
    txt_to_npy(receptor_path)
    os.rename(os.path.join(receptor_path,'mol_1.npy'),receptor_gnina)

if not os.path.exists(os.path.join(args.output,'gninatypes')):
    os.makedirs(os.path.join(args.output,'gninatypes'))
    
os.system('gninatyper/build/gninatyper.out '+docking_file+' '+os.path.join(args.output,'gninatypes')+' mol 1')
t4 = time.time()
txt_to_npy(os.path.join(args.output,'gninatypes'))
t5 = time.time()

# CNN rescoring
cnn_rescoring(receptor_gnina, args.output)
t6 = time.time()

# Calc joined score (CNN + smina)
final_score = joined_score(docking_file,os.path.join(args.output,'cnn_scores.npy'))

# Remove gninatypes and compress docking output
shutil.rmtree(os.path.join(args.output,'gninatypes'))

for f in os.listdir(args.output):
  if f.endswith(".sdf"):
      with open(os.path.join(args.output,f), 'rb') as f_in, gzip.open(os.path.join(args.output,f+'.gz'), 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
      os.remove(os.path.join(args.output,f))

#print('Docking = {:.3f}s'.format(t2-t1))
#print('Sdf filtering = {:.3f}s'.format(t3-t2))
#print('Gnina features = {:.3f}s'.format(t4-t3))
#print('Txt to npy = {:.3f}s'.format(t5-t4))
#print('Network prediction = {:.3f}s'.format(t6-t5))
#print('Final scoring = {:.3f}s'.format(time.time()-t6))
print('Total time = {:.3f}s'.format(time.time()-t1))
