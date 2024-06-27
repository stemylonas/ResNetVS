# ResNetVS
A deep learning-based scoring scheme for virtual screening of small compounds on a target protein (https://doi.org/10.1109/BIBE50027.2020.00030)

Setup
---------------

Experiments were conducted on an Ubuntu 18.04 machine with Python 3.6.9 and CUDA 10.0 

1) Install dependencies
```
sudo apt-get update && apt-get install python3-venv, p7zip, swig, libopenbabel-dev, libboost-all-dev, g++
```
2) Clone this repository
```
git clone https://github.com/stemylonas/ResNetVS
cd ResNetVS
```
3) Create environment and install python dependencies
```
python3 -m venv venv --prompt VScreen
source venv/bin/activate
pip install -r requirements.txt
```
4) Compile gninatyper (originally from https://github.com/gnina/gnina)
```
# tested with g++ 7.5.0
mkdir gninatyper/build
g++ -std=c++11 gninatyper/gninatyper.cpp gninatyper/obmolopener.cpp gninatyper/atom_constants.cpp 
-o gninatyper/build/gninatyper.out -I /usr/include/openbabel-2.0 -I gninatyper -I/usr/local/cuda/include \
-L /usr/lib/openbabel/2.3.2 -lopenbabel -lboost_filesystem -lboost_system -lboost_iostreams
```
5) Download pretrained model
```
pip install gdown
gdown 1Cfi68-VvNeh65xdTypy6xAfNhVH2PKw5
7z e screening_model.7z -o./models
rm screening_model.7z
```
6) Download smina (from https://sourceforge.net/projects/smina/)
```
wget https://sourceforge.net/projects/smina/files/smina.static -P ./smina
chmod +x smina/smina.static
```

Usage example
---------------

```
python run.py -r test_data/spike_mutation_aligned.pdbqt -rH test_data/spike_mutation_aligned_H.pdbqt \
-l test_data/test.sdf -kb test_data/ace2.pdb -o results
```

The method requires an input receptor in two files of .pdbqt format (with and without hydrogens respectively) and an .sdf file of small molecules.
For more input options, check the arguments of 'run.py'.