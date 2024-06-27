/*
 * gninatyper.cpp
 *
 *  Created on: Jun 7, 2016
 *      Author: dkoes
 *
 *  Converts a (single) molecule into a binary file of x,y,z,smina atom type (NOT cnn types)
 *
 *  Modified by smylonas
 *  The output is saved now in txt format (reason --> was unable to load with python the binary files)
 */

#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>
#include <openbabel/oberror.h>

#include "atom_type.h"
#include "obmolopener.h"


using namespace std;
using namespace boost;
using namespace OpenBabel;

struct atom_info {
	float x, y, z;
	int type;

	atom_info(): x(0), y(0), z(0), type(-1) {}
	atom_info(float X, float Y, float Z, int T): x(X), y(Y), z(Z), type(T) {}
};

int main(int argc, char *argv[])
{
	OpenBabel::obErrorLog.StopLogging();
   // cout << 'A' << endl;
	if(argc < 2) {
		cerr << "Need input (and output) file." << "\n";
		exit(-1);
	}

	OBConversion conv;
	obmol_opener opener;
	opener.openForInput(conv, argv[1]);
    //cout << 'B' << endl;
	if(argc >= 3)
	{
		if(algorithm::ends_with(argv[2],".txt"))
		{
			ofstream out(argv[2]);
			if(!out) {
				cerr << "Error opening output file " << argv[2] << "\n";
				exit(1);
			}
			//convert only the first molecule
			OBMol mol;
			conv.Read(&mol);
			if(mol.NumAtoms() == 0) {
				cerr << "Problem reading molecule " << argv[1] << "\n";
				exit(1);
			}
			mol.AddHydrogens();
			FOR_ATOMS_OF_MOL(a, mol)
			{
				smt t = obatom_to_smina_type(*a);
				//atom_info ainfo(a->x(), a->y(), a->z(), t);
				//out.write((char*)&ainfo, sizeof(ainfo));
				out << a->x() << " " << a->y() << " " << a->z() << " " << int(t) << endl;
			}
		}
		else
		{
			//convert all molecules, generating output file names using provided base name
            //cout << 'A' << endl;
			filesystem::path p(argv[1]);
			if (algorithm::ends_with(argv[1], ".gz"))
				p.replace_extension("");
			bool issdf = p.extension() == ".sdf";
            //cout << 'B' << endl;
			string base_path(argv[2]);
			string prefix(argv[3]);
			int cnt = stoi(argv[4]);
			OBMol mol;
			std::istream* in = conv.GetInStream();
           // cout << 'C' << endl;
			while (*in)
			{
				while (conv.Read(&mol))
				{
					mol.AddHydrogens();
					string outname = base_path+"/" + prefix+ "_" + lexical_cast<string>(cnt) + ".txt";
					ofstream out(outname.c_str());
					if (!out)
					{
						cerr << "Error opening output file " << outname << "\n";
						exit(1);
					}
					FOR_ATOMS_OF_MOL(a, mol)
					{
						smt t = obatom_to_smina_type(*a);
						//atom_info ainfo(a->x(), a->y(), a->z(), t);
						//out.write((char*)&ainfo, sizeof(ainfo));
						out << a->x() << " " << a->y() << " " << a->z() << " " << int(t) << endl;
					}
					out.close();
					cnt++;
				}
				if (issdf && *in)
				{ //tolerate molecular errors
					string line;
					while (getline(*in, line))
					{
						if (line == "$$$$")
							break;
					}
					if (*in) cerr << "Encountered invalid molecule " << cnt << "; trying to recover\n";
				}
			}
		}
	}
	else
	{
		//if only input file is specified, auto generate output file name and
		//also handle multiple molecules
		filesystem::path p(argv[1]);
		if(algorithm::ends_with(argv[1],".gz"))
			p.replace_extension("");
		//strip extension
		bool issdf = p.extension() == ".sdf";
		p.replace_extension("");
		boost::unordered_map<string, int> molcnts;
		OBMol mol;
		string name;
		int cnt = 0;
		std::istream* in = conv.GetInStream();

		while(*in) {
      while(conv.Read(&mol)) {
        mol.AddHydrogens();
        name = mol.GetTitle();
        if(name.length() == 0) name = p.string();
        if(molcnts.count(name) == 0) molcnts[name] = 0;
        string outname = name + "_" + lexical_cast<string>(molcnts[name]) + ".txt";
        molcnts[name]++;
        ofstream out(outname.c_str());

        FOR_ATOMS_OF_MOL(a, mol)
        {
          smt t = obatom_to_smina_type(*a);
          //atom_info ainfo(a->x(), a->y(), a->z(), t);
          //out.write((char*)&ainfo, sizeof(ainfo));
	  out << a->x() << " " << a->y() << " " << a->z() << " " << int(t) << endl;
        }
        out.close();
        cnt++;
      }

      if(issdf && *in) { //tolerate molecular errors
        string line;
        while(getline(*in, line)) {
          if(line == "$$$$")
            break;
        }
        if(*in) cerr << "Encountered invalid molecule after " << name << "; trying to recover\n";
      }
		}
	}
    

}
