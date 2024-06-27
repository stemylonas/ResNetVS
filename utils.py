import os, numpy as np
from collections import OrderedDict, defaultdict
from pybel import readfile
from shutil import copy
import operator
import gzip


def get_best_result(lig_name, sol_idx, docked_sdf_file):
    if docked_sdf_file.endswith('gz'):
        with gzip.open(docked_sdf_file,'rt') as f:
            lines = f.readlines()
    else:
        with open(docked_sdf_file,'r') as f:
            lines = f.readlines()
    
    flag = False
    idx = 0
    for l,line in enumerate(lines):
        if lig_name == line[:-1]:
            idx += 1
            if idx==sol_idx:
                l_st = l
                flag = True
        if line[:-1]=='M  END' and flag:
            l_end = l
            break
    
    return lines[l_st:l_end+1]

def close_prot_to_lig(prot_coords,lig_coords,thres):
    for lig_coord in lig_coords:
        dist = min(np.sqrt(np.sum(np.square(prot_coords-lig_coord),axis=1)))
        if dist < thres:
            return True
    return False


def remove_from_sdf(prot_coords,sdf_file,thres=2.0):
    ligands = readfile('sdf',sdf_file)
    
    close = []
    for i,lig in enumerate(ligands):
        lig_coords = [atom.coords for atom in lig]
        close.append(close_prot_to_lig(prot_coords,lig_coords,thres))
    
    lig_to_keep = np.where(close)[0]

    out_file = sdf_file[:-4]+'_filtered.sdf'
    filter_sdf(sdf_file,lig_to_keep,out_file)
    
    return out_file


def filter_sdf(input_sdf,desired_idxs,output_sdf):
    with open(input_sdf,'r') as f:
        lines = f.readlines()
    outlines = []
    idx = 0
    start = 0
    for i,line in enumerate(lines):
        if '$$$$' in line:
            if idx in desired_idxs:
                outlines.extend(lines[start:i+1])
            idx += 1
            start = i+1
         
    with open(output_sdf,'w') as f_out:
        f_out.writelines(outlines)
        
        
def filter_sdf_names(input_sdf,names):
    if input_sdf.endswith('gz'):
        with gzip.open(input_sdf,'rt') as f:
            lines = f.readlines()
    else:
        with open(input_sdf,'r') as f:
            lines = f.readlines()
    outlines = []
    start = 0
    for i,line in enumerate(lines):
        if '$$$$' in line:
            if lines[start][:-1] in names:
                outlines.extend(lines[start:i+1])
            start = i+1
    
    return outlines


def get_sdf_scores(input_sdf):
    scores = OrderedDict()
    if input_sdf.endswith('gz'):
        with gzip.open(input_sdf,'rt') as f:
            lines = f.readlines()
    else:
        with open(input_sdf,'r') as f:
            lines = f.readlines()
    name = lines[0].strip()
    scores[name] = []
    for l,line in enumerate(lines[:-1]):  # avoid last $$$$
        if 'minimizedAffinity' in line:
            scores[name].append(float(lines[l+1]))
        elif '$$$$' in line:
            name = lines[l+1].strip()
            if name not in scores:
                scores[name] = []
    
    return scores


def txt_to_npy(path):
    for f in os.listdir(path):
        if f.endswith('.txt'):
            txt_file = os.path.join(path,f)
            data = np.loadtxt(txt_file,dtype=np.float16)
            np.save(txt_file.rsplit('.')[0]+'.npy',data)
            os.remove(txt_file)
            del data


def read_final_score(input_file):
    with open(input_file,'r') as f:
        lines = f.readlines()
    lines = [line for line in lines if line!='\n']
    score = dict()
    for line in lines:
        parts = line.split(':',2)
        score[parts[2].strip()] = float(parts[0])
    return score


def compare_conform(paths):  # list of paths
    all_scores = defaultdict(list)
    for path in paths:
        score = read_final_score(os.path.join(path,'final_score.txt'))
        for key,val in score.items():
            all_scores[key].append(val)
    
    for key,val in all_scores.items():
        if len(val)!=len(paths):
            for i in range(len(paths)-len(val)):
                all_scores[key].append(0.0)
    
    return all_scores
     

def write_joined_conform_scores(paths,out_path,case):  # top1, top2, top3
    idx = int(case[-1])
    scores_dict = compare_conform(paths)
    sorted_scores = sorted(scores_dict.items(), key=lambda elem: sorted(elem[1])[len(paths)-idx], reverse=True)
    with open(os.path.join(out_path,'sorting_'+case+'.txt'), 'w') as f:        
        for lig_name,val in sorted_scores:
            val_str = '('
            for v in val:
                val_str += " {0:.3f}".format(v)
            val_str += ')'
            best_val = sorted(val)[len(paths)-idx]
            f.write(val_str+' : '+str(best_val)+' : '+lig_name+'\n')
  
    
def read_from_joined_file(input_file):
    with open(input_file,'r') as f:
        lines = f.readlines()
    score = OrderedDict()
    for line in lines:
        parts = line.split(':',2)
        score[parts[2].strip()] = float(parts[1])
    return score


def select_best_results(joined_score_file,top_n):
    with open(joined_score_file,'r') as f:
        lines = f.readlines()
    return set([line.split(':')[-1] for line in lines[:top_n]])


def merge_micro(paths,out_path):
    final_scores1 = read_from_joined_file(os.path.join(paths[0],'sorting_conf2.txt'))
    for path in paths[1:]:
        final_scores2 = read_from_joined_file(os.path.join(path,'sorting_conf2.txt'))
        final_scores1.update(final_scores2)
    sorted_scores = sorted(final_scores1.items(), key=lambda elem: elem[1], reverse=True)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path,'sorting_conf2.txt'), 'w') as f:        
        for lig_name,val in sorted_scores:
            f.write(': '+str(val)+' : '+lig_name+'\n')
    

def hetatm_to_atom(input_path):
    for f in os.listdir(input_path):
        with open(os.path.join(input_path,f),'r') as ff:
            lines=ff.readlines()
        lines=[line.replace('HETATM','ATOM  ') for line in lines]
        with open(os.path.join(input_path,f),'w') as ff:
            ff.writelines(lines)


def get_ligand_names_from_sdf(sdf_file):
    with open(sdf_file,'r') as f:
        lines = f.readlines()
    ligands = [lines[0][:-1]]
    for l,line in enumerate(lines[:-2]):
        if line[:4]=='$$$$':
            ligands.append(lines[l+1][:-1])
    
    return ligands


def filter_best_results(score_file,pdb_path,out_paths,T):
    scores_dict = read_from_joined_file(score_file)
    sorted_scores = sorted(scores_dict.items(), key=operator.itemgetter(1), reverse=True)
    ligands = [f[:-4] for f in os.listdir(pdb_path)]
    if not os.path.exists(out_paths[0]):
        os.makedirs(out_paths[0])
    if not os.path.exists(out_paths[1]):
        os.makedirs(out_paths[1])
    for lig,val in sorted_scores:
        if lig not in ligands:
            continue
        if val>=T:
            copy(os.path.join(pdb_path,lig+'.pdb'),os.path.join(out_paths[0],lig+'.pdb'))
        else:
            copy(os.path.join(pdb_path,lig+'.pdb'),os.path.join(out_paths[1],lig+'.pdb'))       
