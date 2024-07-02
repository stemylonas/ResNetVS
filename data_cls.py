import numpy as np

        
class MoleculeComplex:
    def __init__(self,rec_input,lig_input):
        if type(rec_input)==str:
            rec = np.load(rec_input)
        else:
            rec = np.copy(rec_input)
        if type(lig_input)==str:
            lig = np.load(lig_input)
        else:
            lig = np.copy(lig_input)
            
        self.coords = np.concatenate((rec[:,:-1],lig[:,:-1]))

        self.center = np.average(self.coords[len(rec):,:],axis=0)
        self.types = np.concatenate((rec[:,-1],lig[:,-1]))
        self.types = self.types.astype(int)
        self.is_lig = np.concatenate((np.zeros(len(rec),dtype=np.int8),np.ones(len(lig),dtype=np.int8)))   # 0 for receptor, 1 for ligand
        
    def rotate(self):
        ax,ay,az = np.random.uniform(-np.pi,np.pi,3)
        self.coords -= self.center
        coords1 = np.copy(self.coords)
        coords1[:,1] = np.cos(ax)*self.coords[:,1] - np.sin(ax)*self.coords[:,2]
        coords1[:,2] = np.sin(ax)*self.coords[:,1] + np.cos(ax)*self.coords[:,2]
        coords2 = np.copy(coords1)
        coords2[:,0] = np.cos(ay)*coords1[:,0] + np.sin(ay)*coords1[:,2]
        coords2[:,2] = -np.sin(ay)*coords1[:,0] + np.cos(ay)*coords1[:,2]
        coords3 = np.copy(coords2)
        coords3[:,0] = np.cos(az)*coords2[:,0] - np.sin(az)*coords2[:,1]
        coords3[:,1] = np.sin(az)*coords2[:,0] + np.cos(az)*coords2[:,1]
        self.coords = coords3 + self.center
        
    def translate(self,limit):
        offset = np.random.uniform(-limit,limit,3)
        self.coords += offset
        self.center += offset


class Grid:
    def __init__(self,atom_typing,cube_size,cell_dim,nAtomTypes):
        self.atom_typing = atom_typing
        self.cell_dim = cell_dim
        self.cube_size = cube_size
        self.grid_size = int(cube_size/cell_dim)
        self.nAtomTypes = nAtomTypes
    
    def create_grid(self,mol):
        grid_limits_low = mol.center - float(self.cube_size)/2
        grid_limits_up = mol.center + float(self.cube_size)/2
        feats = np.zeros((self.grid_size,self.grid_size,self.grid_size,self.nAtomTypes),dtype=np.float32)
        for i,p in enumerate(mol.coords):
            radius = ATOM_TYPES[mol.types[i]][1]
            if np.any(p+radius<grid_limits_low) or np.any(p-radius>grid_limits_up):   #  discard atoms out of the grid
                continue
            if ATOM_TYPES[mol.types[i]][2+mol.is_lig[i]]:  # ignore atoms of specific type
                continue
            idx_low = np.maximum(np.floor((p-radius-grid_limits_low)/self.cell_dim),0)
            idx_up = np.minimum(np.floor((p+radius-grid_limits_low)/self.cell_dim),self.grid_size-1)
            idx_low = idx_low.astype(int)
            idx_up = idx_up.astype(int)
            grid_type_idx = ATOM_TYPES[mol.types[i]][4+mol.is_lig[i]]
            #t1=time.time()
            if self.atom_typing == 'boolean':
                feats[idx_low[0]:idx_up[0]+1,idx_low[1]:idx_up[1]+1,idx_low[2]:idx_up[2]+1,grid_type_idx] = 1
#            elif self.atom_typing == 'continuous':  
#                for x_idx in range(idx_low[0],idx_up[0]+1):
#                    for y_idx in range(idx_low[1],idx_up[1]+1):
#                        for z_idx in range(idx_low[2],idx_up[2]+1):
#                            cell_center = np.array([x_idx+0.5,y_idx+0.5,z_idx+0.5])*self.cell_dim + grid_limits_low
#                            d = np.sqrt(sum((cell_center-p)**2))
#                            feats[x_idx,y_idx,z_idx,grid_type_idx] += self.density(d,radius)
            elif self.atom_typing == 'continuous': 
                grid_idxs = np.mgrid[idx_low[0]:idx_up[0]+1,idx_low[1]:idx_up[1]+1,idx_low[2]:idx_up[2]+1]
                cell_centers = (np.transpose(grid_idxs,(1,2,3,0)) + 0.5)*self.cell_dim + grid_limits_low
                d = np.sqrt(np.sum((cell_centers-p)**2,axis=3)) + 0.00001
                #t4=time.time()
                #densities1 = np.reshape(np.array(map(lambda x: self.density_v2(x,radius),d.flatten())),d.shape)
                densities = 1-np.exp(-(radius/d)**12)
               # if np.sum(densities-densities1)>0:
                #    print "error found", np.sum(densities-densities1)
                #t7=time.time()
                #densities = np.reshape(np.array([self.density(x,radius) for x in d.flatten()]),d.shape)
                feats[idx_low[0]:idx_up[0]+1,idx_low[1]:idx_up[1]+1,idx_low[2]:idx_up[2]+1,grid_type_idx] += densities

                #print t7-t4
        return feats
    
    def density(self,d,r):
        if d<r:
            return np.exp(-2*d*d/r/r)
        elif d<1.5*r:
            return 4*np.exp(-2)*(d/r)**2 - 12*np.exp(-2)*d/r + 9*np.exp(-2)
        else:
            return 0
    
    # def density_v2(self,d,r):
    #    return 1-np.exp(-(r/d)**12)
 


ATOM_TYPES = [['Hydrogen',1.0,True,True,np.inf,np.inf],
              ['PolarHydrogen',1.0,True,True,np.inf,np.inf],
              ['AliphaticCarbonXSHydrophobe',2.0,False,False,0,12],
              ['AliphaticCarbonXSNonHydrophobe',2.0,False,False,1,13],
              ['AromaticCarbonXSHydrophobe',2.0,False,False,2,14],
              ['AromaticCarbonXSNonHydrophobe',2.0,False,False,3,15],
              ['Nitrogen',1.75,False,False,4,16],
              ['NitrogenXSDonor',1.75,False,False,5,17],
              ['NitrogenXSDonorAcceptor',1.75,False,False,6,18],
              ['NitrogenXSAcceptor',1.75,False,False,7,19],
              ['Oxygen',1.6,True,True,np.inf,np.inf],
              ['OxygenXSDonor',1.6,True,True,np.inf,np.inf],
              ['OxygenXSDonorAcceptor',1.6,False,False,8,20],
              ['OxygenXSAcceptor',1.6,False,False,9,21],
              ['Sulfur',2.0,False,False,10,22],
              ['SulfurAcceptor',2.0,True,True,np.inf,np.inf],
              ['Phosphorus',2.1,True,False,np.inf,23],
              ['Fluorine',1.55,True,False,np.inf,24],
              ['Chlorine',2.05,True,False,np.inf,25],
              ['Bromine',2.17,True,False,np.inf,26],
              ['Iodine',2.36,True,False,np.inf,27],
              ['Magnesium',0.65,True,True,np.inf,np.inf],
              ['Manganese',0.65,True,True,np.inf,np.inf],
              ['Zinc',0.74,True,True,np.inf,np.inf],
              ['Calcium',0.99,True,True,np.inf,np.inf],
              ['Iron',0.65,True,True,np.inf,np.inf],
              ['GenericMetal',1.2,False,True,11,np.inf],
              ['Boron',2.04,True,True,np.inf,np.inf]]
