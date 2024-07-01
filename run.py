import os, argparse, time, shutil, gzip
from docking import SminaDocking
from scoring import cnn_rescoring, joined_score
from utils import txt_to_npy


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


if not os.path.exists(args.output):
    os.makedirs(args.output)
t1 = time.time()

# Docking

docking = SminaDocking(args.receptor, args.ligand, args.output, args.known_binder, args.residues)
docking.run()

# Get gnina features
    
receptor_gnina = args.receptor.rsplit('.',1)[0] + '_gninatypes.npy'

if not os.path.exists(receptor_gnina):
    receptor_path = args.receptor.rsplit('/',1)[0]
    os.system('gninatyper/build/gninatyper.out '+args.receptor+' '+receptor_path+' mol 1')
    txt_to_npy(receptor_path)
    os.rename(os.path.join(receptor_path,'mol_1.npy'),receptor_gnina)

if not os.path.exists(os.path.join(args.output,'gninatypes')):
    os.makedirs(os.path.join(args.output,'gninatypes'))
    
os.system('gninatyper/build/gninatyper.out '+docking.docking_file+' '+os.path.join(args.output,'gninatypes')+' mol 1')
t4 = time.time()
txt_to_npy(os.path.join(args.output,'gninatypes'))
t5 = time.time()

# CNN rescoring
cnn_rescoring(receptor_gnina, args.output)
t6 = time.time()

# Calc joined score (CNN + smina)
final_score = joined_score(docking.docking_file,os.path.join(args.output,'cnn_scores.npy'))

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
