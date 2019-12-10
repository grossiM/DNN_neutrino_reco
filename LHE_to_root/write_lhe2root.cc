// -*- C++ -*-
//2019 M.Grossi - J. Novak
// single_file is a simple converter, which reads from LHE file format and writes to 
// root format. It keeps track of initial and final state particles. 
//At most two W bosons are deduced from lepton neutrino pair. No information about parents is recorded.
//
// Run the program by:
// source setEnv.sh
// make
// ./write_lhe2root gridpack outfile (example /write_lhe2root mu_ewk_semilept_lsf mu_ewk_semilept_lsf_out)
//hadd mu_ewk_semilept_lsf_merged_lhe.root gen*.root
// gridpack is the folder containing Phantom generations
//
// Warning: outfile without appendix (.root) !!!
//

#include "LHEF.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
//#include "TLorentzVector.h"
//#include "TVector2.h"
//#include "TVectorD.h"

#include <boost/random.hpp>
#include <cstdlib>
#include <string>
#include <sstream>
#include <cassert>
#include <iostream>
#include <dirent.h>

using namespace std;


int single_file(string infile, string outfile) {

  outfile = outfile + "_lhe.root";
  //variable delcaration
  int flavour;
  //electron
  vector<double> el_px; //el_px
  vector<double> el_py;
  vector<double> el_pz;
  vector<double> el_E;
  //muon
  vector<double> mu_px;
  vector<double> mu_py;
  vector<double> mu_pz;
  vector<double> mu_E;
  //electron neutrino
  vector<double> v_el_px;
  vector<double> v_el_py;
  vector<double> v_el_pz;
  vector<double> v_el_E;
  //muon neutrino
  vector<double> v_mu_px;
  vector<double> v_mu_py;
  vector<double> v_mu_pz;
  vector<double> v_mu_E;
  //w boson decays to electron
  /*vector<double> W_el_px;
  vector<double> W_el_py;
  vector<double> W_el_pz;
  vector<double> W_el_E;
  //w boson decays to muon
  vector<double> W_mu_px;
  vector<double> W_mu_py;
  vector<double> W_mu_pz;
  vector<double> W_mu_E;*/

  //check if to keep
  //TVector2 MET;
  //initial quarks
  //vector< TLorentzVector > initial_q;
  vector<double> q_init_px;
  vector<double> q_init_py;
  vector<double> q_init_pz;
  vector<double> q_init_E;

  //final quarks
  //vector< TLorentzVector > final_q;
  vector<double> q_fin_px;
  vector<double> q_fin_py;
  vector<double> q_fin_pz;
  vector<double> q_fin_E;

  //vector< TLorentzVector > w_boson;//
  vector<double> w_bos_px;
  vector<double> w_bos_py;
  vector<double> w_bos_pz;
  vector<double> w_bos_E;

  int sign;
//////////////////////////////////////////////////////////////////////////////////
// TTree to save jets
  TFile * partonfile = new TFile(outfile.c_str(), "RECREATE");
  TTree * partontree = new TTree("tree", "tree");

  //branch declaration
  partontree->Branch("flavour", &flavour);
  //branch electron
  partontree->Branch("el_px", &el_px);//el_px
  partontree->Branch("el_py", &el_py);
  partontree->Branch("el_pz", &el_pz);
  partontree->Branch("el_E", &el_E);

  //branch muon
  partontree->Branch("mu_px", &mu_px);
  partontree->Branch("mu_py", &mu_py);
  partontree->Branch("mu_pz", &mu_pz);
  partontree->Branch("mu_E", &mu_E);

  //branch electron neutrino
  partontree->Branch("v_el_px", &v_el_px);//name of the vector should be the same v_el_px
  partontree->Branch("v_el_py", &v_el_py);
  partontree->Branch("v_el_pz", &v_el_pz);
  partontree->Branch("v_el_E", &v_el_E);

  //branch muon neutrino
  partontree->Branch("v_mu_px", &v_mu_px);//cambiare con notazione vettore
  partontree->Branch("v_mu_py", &v_mu_py);
  partontree->Branch("v_mu_pz", &v_mu_pz);
  partontree->Branch("v_mu_E", &v_mu_E);
  
  //branch w boson
  partontree->Branch("w_bos_px", &w_bos_px);
  partontree->Branch("w_bos_py", &w_bos_py);
  partontree->Branch("w_bos_pz", &w_bos_pz);
  partontree->Branch("w_bos_E", &w_bos_E);

  //branch initial quarks
  partontree->Branch("q_init_px", &q_init_px);
  partontree->Branch("q_init_py", &q_init_py);
  partontree->Branch("q_init_pz", &q_init_pz);
  partontree->Branch("q_init_E", &q_init_E);

  //branch final quarks
  partontree->Branch("q_fin_px", &q_fin_px);
  partontree->Branch("q_fin_py", &q_fin_py);
  partontree->Branch("q_fin_pz", &q_fin_pz);
  partontree->Branch("q_fin_E", &q_fin_E);

  //branch w boson decays to mu
  /*partontree->Branch("W_mu_px", &W_mu_px);
  partontree->Branch("W_mu_py", &W_mu_py);
  partontree->Branch("W_mu_pz", &W_mu_pz);
  partontree->Branch("W_mu_E", &W_mu_E);

  //branch w w boson decays to electron
  partontree->Branch("W_el_px", &W_el_px);
  partontree->Branch("W_el_py", &W_el_py);
  partontree->Branch("W_el_pz", &W_el_pz);
  partontree->Branch("W_el_E", &W_el_E);*/

  //partontree->Branch("MET", &MET);	//check
  partontree->Branch("sign", &sign);//check
													
  // Init readers and writers
  LHEF::Reader* reader = 0;
  if (infile == "-") {
    reader = new LHEF::Reader(cin);
  } else {
    reader = new LHEF::Reader(infile);
  }

  int ev_num = 0;

  // Event loop
  while (reader->readEvent()) {

    double p_x=0;//check
    double p_y=0;//check

    bool have_elec = false;
    bool have_muon = false;

    sign = 1;
    flavour ==-1;

    if (ev_num %1000 == 0) cout << "events: " << ev_num << endl;
      
    // NUP number of particles in event
    //for loop on particles
    for (int i = 0; i < reader->hepeup.NUP; ++i) {
      //ISTUP[i] == 1 final state parton
      if (reader->hepeup.ISTUP[i] == 1) {
          // IDUP[i]) == 13 mu- . -13 mu +
        if (abs(reader->hepeup.IDUP[i]) == 13){
          if (reader->hepeup.IDUP[i]==13) sign = -1;
          else if (sign==-1) cout<<"Sign error"<<endl;
          //diventa muon_px.push_back(reader->hepeup.PUP[i][0])
          mu_px.push_back(reader->hepeup.PUP[i][0]);
          mu_py.push_back(reader->hepeup.PUP[i][1]);
          mu_pz.push_back(reader->hepeup.PUP[i][2]);
          mu_E.push_back(reader->hepeup.PUP[i][3]);
          have_muon = true;
                }
        
        if (abs(reader->hepeup.IDUP[i]) == 11){
          //// IDUP[i]) == 11 e-
          if (reader->hepeup.IDUP[i]==11) sign = -1;
          else if (sign==-1) cout<<"Sign error"<<endl;
          el_px.push_back(reader->hepeup.PUP[i][0]);
          el_py.push_back(reader->hepeup.PUP[i][1]);
          el_pz.push_back(reader->hepeup.PUP[i][2]);
          el_E.push_back(reader->hepeup.PUP[i][3]);
          have_elec = true;
                }

        if (abs(reader->hepeup.IDUP[i]) == 14){
          //// IDUP[i]) == 14 nu mu
          v_mu_px.push_back(reader->hepeup.PUP[i][0]);
          v_mu_py.push_back(reader->hepeup.PUP[i][1]);
          v_mu_pz.push_back(reader->hepeup.PUP[i][2]);
          v_mu_E.push_back(reader->hepeup.PUP[i][3]);
            p_x += reader->hepeup.PUP[i][0];//check
            p_y += reader->hepeup.PUP[i][1];//check
                  }

        if (abs(reader->hepeup.IDUP[i]) == 12){
          //// IDUP[i]) == 12 nu e
          v_el_px.push_back(reader->hepeup.PUP[i][0]);
          v_el_py.push_back(reader->hepeup.PUP[i][1]);
          v_el_pz.push_back(reader->hepeup.PUP[i][2]);
          v_el_E.push_back(reader->hepeup.PUP[i][3]);
            p_x += reader->hepeup.PUP[i][0];//check
            p_y += reader->hepeup.PUP[i][1];//check
                  }

        if (abs(reader->hepeup.IDUP[i]) == 24){
          //// IDUP[i]) == 24 W 
          w_bos_px.push_back(reader->hepeup.PUP[i][0]);
          w_bos_py.push_back(reader->hepeup.PUP[i][1]);
          w_bos_pz.push_back(reader->hepeup.PUP[i][2]);
          w_bos_E.push_back(reader->hepeup.PUP[i][3]);
                  }

        if (abs(reader->hepeup.IDUP[i]) < 7) {
          //// IDUP[i]) == from 1 to 6 all quarks
          q_fin_px.push_back(reader->hepeup.PUP[i][0]);
          q_fin_py.push_back(reader->hepeup.PUP[i][1]);
          q_fin_pz.push_back(reader->hepeup.PUP[i][2]);
          q_fin_E.push_back(reader->hepeup.PUP[i][3]);
                  }
      } // end of final state particles loop


      if (reader->hepeup.ISTUP[i] == -1) {
        ////ISTUP[i] == -1 initial state parton
          if (abs(reader->hepeup.IDUP[i]) < 7){
            q_init_px.push_back(reader->hepeup.PUP[i][0]);
            q_init_py.push_back(reader->hepeup.PUP[i][1]);
            q_init_pz.push_back(reader->hepeup.PUP[i][2]);
            q_init_E.push_back(reader->hepeup.PUP[i][3]);
                    }
              }

      }	 // for loop on particles

      if (have_elec && (! have_muon)){
          flavour = 1;
      }else if((!have_elec) && have_muon){
          flavour = 0;
      }else{
          flavour = -1;
          // std::cout << "ERROR! Boh electron and muon in the event!" << std::endl;
      }
      //delete branch etc with w_el and muon

      //MET = TVector2(p_x, p_y);

      partontree->Fill();
 
      //erase of all vectors

      mu_px.erase(mu_px.begin(),mu_px.end());
      mu_py.erase(mu_py.begin(),mu_py.end());
      mu_pz.erase(mu_pz.begin(),mu_pz.end());
      mu_E.erase(mu_E.begin(),mu_E.end());

      el_px.erase(el_px.begin(),el_px.end());
      el_py.erase(el_py.begin(),el_py.end());
      el_pz.erase(el_pz.begin(),el_pz.end());
      el_E.erase(el_E.begin(),el_E.end());

      v_mu_px.erase(v_mu_px.begin(),v_mu_px.end());
      v_mu_py.erase(v_mu_py.begin(),v_mu_py.end());
      v_mu_pz.erase(v_mu_pz.begin(),v_mu_pz.end());
      v_mu_E.erase(v_mu_E.begin(),v_mu_E.end());

      v_el_px.erase(v_el_px.begin(),v_el_px.end());
      v_el_py.erase(v_el_py.begin(),v_el_py.end());
      v_el_pz.erase(v_el_pz.begin(),v_el_pz.end());
      v_el_E.erase(v_el_E.begin(),v_el_E.end());

      q_fin_px.erase(q_fin_px.begin(),q_fin_px.end()); 
      q_fin_py.erase(q_fin_py.begin(),q_fin_py.end()); 
      q_fin_pz.erase(q_fin_pz.begin(),q_fin_pz.end()); 
      q_fin_E.erase(q_fin_E.begin(),q_fin_E.end()); 

      q_init_px.erase(q_init_px.begin(),q_init_px.end());
      q_init_py.erase(q_init_py.begin(),q_init_py.end());
      q_init_pz.erase(q_init_pz.begin(),q_init_pz.end());
      q_init_E.erase(q_init_E.begin(),q_init_E.end());


      ev_num++;     

  } // End of events loop

  partontree->Write();
  partonfile->Close();

  delete reader;

  return ev_num;
}

int main(int argc, char** argv) {

  // Look for a help argument
  for (int i = 1; i < argc; ++i) {
    const string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      cout << argv[0] << ": an LHEF to HepMC event format converter" << endl;
      cout << "Usage: " << argv[0] << "[<infold> <outfold>]" << endl;
      exit(0);
    }
  }

  // Choose input and output files from the command line
  string infold("-"), outfold("-");
  if (argc == 3) {
    infold = argv[1];
    outfold = argv[2];
  } else if (argc == 2 || argc > 3) {
    cerr << "Usage: " << argv[0] << "[<infold> <outfold>]" << endl;
    exit(1);
  }

  string infofile = outfold + "/info.root";

  int nevts;
  int njob = 0;

  TTree* t = new TTree("job_info","job_info");
  t->Branch("job_number",&njob);
  t->Branch("n_events", &nevts);

  DIR* dir;
  dirent* pdir;
  dir = opendir(infold.c_str());

  while(pdir = readdir(dir)){
    njob++;
    string fold = pdir->d_name;
    if (fold.find("gen")==string::npos) continue;
    if (fold.find("gendir")!=string::npos ) continue;
    if (fold.find("submitfile")!=string::npos ) continue;
    cout<<pdir->d_name<<endl;
    string full_foldname = string(infold) + "/" + string(fold);

    DIR* dir1;
    dirent* pdir1;
    dir1 = opendir(full_foldname.c_str());

    while(pdir1 = readdir(dir1)){
      string file = pdir1->d_name;
      if (file.find("phamom")==string::npos) continue;
      string full_filename = string(full_foldname) + "/" + string(file);
      string full_outname = string(outfold) + "/" + string(fold);
      nevts = single_file(full_filename, full_outname);
    }
    t->Fill();
  }

  TFile* f = new TFile(infofile.c_str(),"recreate");
  t->Write();
  f->Close();

  return EXIT_SUCCESS;
}

