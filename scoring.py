import os
import numpy as np
import operator
import tensorflow as tf
from tensorflow.contrib import slim

from resnet_3d import resnet_arg_scope, resnet_v1_18
from data_cls import MoleculeComplex, Grid
from utils import get_sdf_scores


def fused_score(smina_val,cnn_val,t_norm):
    if t_norm=='prod':
        return 1/(1+np.exp(-1.465*(abs(smina_val)-6))) * cnn_val
    elif t_norm=='min':
        return np.minimum(1/(1+np.exp(-1.465*(abs(smina_val)-6))),cnn_val)

def aggregate_scores(fused_val,s_norm):
    if s_norm=='sum':
        #return np.average(fused_val)
        return np.sum(fused_val[fused_val>0.5])
    elif s_norm=='max':
        max_idx = np.argmax(fused_val)
        return fused_val[max_idx], max_idx+1

def joined_score(smina_output, cnn_output, t_norm='prod',s_norm='max'):
    #t_norm = 'prod'   # prod or min
    #s_norm = 'max'    # sum or max
    
    smina_scores = get_sdf_scores(smina_output)
    
    cnn_scores = np.load(cnn_output)
    
    ligand_scores = dict()
    
    sols_per_lig = [len(v) for v in smina_scores.values()]
    cnn_val_split = np.split(cnn_scores,np.cumsum(sols_per_lig)[:-1])
    
    for lig_idx,key in enumerate(smina_scores):
        smina_val = np.asarray(smina_scores[key])
        cnn_val = cnn_val_split[lig_idx]
        fused_val = fused_score(smina_val,cnn_val,t_norm)
        ligand_scores[key] = aggregate_scores(fused_val,s_norm)
    
    sorted_scores = sorted(ligand_scores.items(), key=operator.itemgetter(1), reverse=True)
    
    with open(smina_output.rsplit('/',1)[0]+'/final_score.tsv', 'w') as f:
        f.write("ligand\tpose_id\tscore\n")
        for lig_val in sorted_scores:
            f.write(lig_val[0]+'\t'+str(lig_val[1][1])+'\t'+'{0:.3f}'.format(lig_val[1][0])+'\n')
        
    return sorted_scores
   
   
def cnn_rescoring(rec_file, base_path):
    nCl = 2
    atom_typing = 'boolean'
    cube_size = 24
    cell_dim = 1
    nAtomTypes = 28
    model_path = 'models'
    gninatypes_path = os.path.join(base_path,'gninatypes')
    
    grid = Grid(atom_typing,cube_size,cell_dim,nAtomTypes)
    
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32,shape=(1,grid.grid_size,grid.grid_size,grid.grid_size,nAtomTypes))
    
    with slim.arg_scope(resnet_arg_scope()):
        net, end_points = resnet_v1_18(inputs, nCl, is_training=False)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer()) 
    saver = tf.train.Saver()
    saver.restore(sess,os.path.join(model_path,'DUDE_3x3_olds_set_oldfold0_49'))
    
    n_poses = len(os.listdir(gninatypes_path))
    out_prob = np.zeros((n_poses,2))
    
    for i in range(n_poses):
        mol = MoleculeComplex(rec_file,os.path.join(gninatypes_path,'mol_'+str(i+1)+'.npy'))
        feats = grid.create_grid(mol)
        output = sess.run(end_points,feed_dict={inputs:np.expand_dims(feats,axis=0)})
        out_prob[i,:] = np.squeeze(output['predictions'])
    
    sess.close()
    
    np.save(os.path.join(base_path,'cnn_scores.npy'),out_prob[:,1])