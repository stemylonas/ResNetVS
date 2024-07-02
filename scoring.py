import os, shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from resnet_3d import resnet_arg_scope, resnet_v1_18
from data_cls import MoleculeComplex, Grid
from utils import txt_to_npy


def fused_score(smina_score, cnn_score, t_norm):
    if t_norm=='prod':
        return smina_score * cnn_score
    elif t_norm=='min':
        return np.minimum(smina_score, cnn_score)

def aggregate_scores(fused_val, s_norm):
    if s_norm=='sum':
        return np.sum(fused_val[fused_val>0.5])
    elif s_norm=='max':
        max_idx = np.argmax(fused_val)
        return fused_val[max_idx], max_idx+1

def smina_scoring(affinity):
    return 1/(1+np.exp(-1.465*(abs(affinity)-6)))

def joined_scoring(smina_affinities, cnn_scores, t_norm='prod', s_norm='max'):
       
    ligand_scores = []
    
    poses_per_lig = [len(v) for v in smina_affinities.values()]
    cnn_scores_per_lig = np.split(cnn_scores,np.cumsum(poses_per_lig)[:-1])
    
    for lig_idx,key in enumerate(smina_affinities):
        affinities_per_lig = np.asarray(smina_affinities[key])
        smina_val = smina_scoring(affinities_per_lig)
        cnn_val = cnn_scores_per_lig[lig_idx]
        fused_val = fused_score(smina_val, cnn_val, t_norm)
        ligand_score, best_pose_idx = aggregate_scores(fused_val, s_norm)
        ligand_scores.append(
                {'name':key, 'score':ligand_score, 'best_pose_idx':best_pose_idx}
            )
    
    sorted_scores = sorted(ligand_scores, key=lambda x: x['score'], reverse=True)
        
    return sorted_scores
   
   
class Rescoring:
    def __init__(self, atom_typing='boolean', cube_size=24, cell_dim=1, nAtomTypes=28):

        self.model = 'models/DUDE_3x3_olds_set_oldfold0_49'
        self.grid = Grid(atom_typing, cube_size, cell_dim, nAtomTypes)
    
    def predict(self, receptor, docking_result, out_dir, batch_size=16):
        receptor_gnina, ligand_gnina_path = self._calc_gninatypes(receptor, docking_result, out_dir)
        
        tf.reset_default_graph()
        grid_size = self.grid.grid_size
        nAtomTypes = self.grid.nAtomTypes
        inputs = tf.placeholder(tf.float32,shape=(None,grid_size,grid_size,grid_size,nAtomTypes))
        
        with slim.arg_scope(resnet_arg_scope()):
            net, end_points = resnet_v1_18(inputs, 2, is_training=False)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        sess.run(tf.global_variables_initializer()) 
        saver = tf.train.Saver()
        saver.restore(sess, self.model)
        
        n_poses = len(os.listdir(ligand_gnina_path))
        n_batches = (n_poses + batch_size - 1) // batch_size
        out_prob = np.zeros(n_poses)
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_poses)
            
            feats = np.zeros((end_idx-start_idx,grid_size,grid_size,grid_size,nAtomTypes))
            for j in range(end_idx-start_idx):
                mol = MoleculeComplex(receptor_gnina, os.path.join(ligand_gnina_path,'mol_'+str(start_idx+j+1)+'.npy'))
                feats[j,:,:,:,:] = self.grid.create_grid(mol)
            
            output = sess.run(end_points,feed_dict={inputs:feats})

            out_prob[start_idx:end_idx] = np.squeeze(output['predictions'], axis=(1,2,3))[:,1]
        
        sess.close()
        
        # Remove ligand gninatypes 
        shutil.rmtree(ligand_gnina_path)
        
        return out_prob
    
    def _calc_gninatypes(self, receptor, docking_result, out_dir):
        receptor_gnina = receptor.rsplit('.',1)[0] + '_gninatypes.npy'
        if not os.path.exists(receptor_gnina):
            receptor_path = receptor.rsplit('/',1)[0]
            self._run_gninatyper(receptor, receptor_path)
            os.rename(os.path.join(receptor_path,'mol_1.npy'), receptor_gnina)
        
        ligand_gnina_path = os.path.join(out_dir,'gninatypes')
        if not os.path.exists(ligand_gnina_path):
            os.makedirs(ligand_gnina_path)
    
        self._run_gninatyper(docking_result, ligand_gnina_path)
        
        return receptor_gnina, ligand_gnina_path
    
    def _run_gninatyper(self, mol_file, out_dir):
        os.system('gninatyper/build/gninatyper.out '+mol_file+' '+out_dir+' mol 1')
        txt_to_npy(out_dir)