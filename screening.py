import os, argparse, time
from docking import SminaDocking
from scoring import Rescoring, joined_scoring


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

if not os.path.exists(args.output):
    os.makedirs(args.output)

# Docking
t1 = time.time()

docking = SminaDocking(args.receptor_H, args.ligand, args.output, args.known_binder, args.residues)
docking.run()
smina_affinities = docking.get_affinities()

# CNN rescoring
t2 = time.time()

cnn_rescoring = Rescoring()
cnn_scores = cnn_rescoring.predict(args.receptor, docking.docking_file, args.output)

# Calc joined score (CNN + smina)
final_scores = joined_scoring(smina_affinities, cnn_scores)

t3 = time.time()

# Write final scores to .tsv
with open(args.output+'/final_score.tsv', 'w') as f:
    f.write("ligand\tpose_id\tscore\n")
    for ligand_score in final_scores:
        f.write(ligand_score['name']+'\t'+str(ligand_score['best_pose_idx'])+'\t'+'{0:.3f}'.format(ligand_score['score'])+'\n')

# compress docking output
docking.compress_output()

print('Docking time = {:.3f}s'.format(t2-t1))
print('Rescoring time = {:.3f}s'.format(t3-t2))
print('Total time = {:.3f}s'.format(time.time()-t1))