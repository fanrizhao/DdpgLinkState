//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//

#include <Statistic.h>
#include <limits>

Statistic *Statistic::inst = 0;



Statistic::Statistic() {

    INI = 50;
    END = 650;
    collect = true;

    Traffic =  vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));
    Routing =  vector<vector<double>  > (100, vector<double>(100));
    Delay =  vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));

    DropsV =  vector<vector<double>  > (100, vector<double>(100, 0));
    drops = 0;
    SendP = vector<vector<long int>>(100, vector<long int>(100,0));
    Trafficlist = vector<vector<double>>(100, vector<double>(100,0));
    trafficFlag = false;

}



Statistic::~Statistic() {

}



Statistic *Statistic::instance() {
    if (!inst)
      inst = new Statistic();
    return inst;
}

void Statistic::setMaxSim(double ms) {
    END = ms;
    SIMTIME = (END.dbl()-INI.dbl())*1000;
}

void Statistic::setRouting(int src, int dst, double r) {
    (Routing)[src][dst] = r;
}

void Statistic::infoTS(simtime_t time) {
    if (time > END and collect) {
        collect = false;
        printStats();
    }
    if (time < INI and not collect)
        collect = true;
}


void Statistic::setDelay(simtime_t time, int src, int dst, double d) {
    if (time > INI and collect){
        (Delay)[src][dst].push_back(d);
       // (SendP)[src][dst]++;
    }


}


void Statistic::setTraffic(simtime_t time, int src, int dst) {
    if (time > INI and collect){
        (SendP)[src][dst]++;
    }
//        (Traffic)[src][dst].push_back(t);

}

void Statistic::setLost(simtime_t time, int n, int p) {
    if (time > INI and collect) {
        drops++;
        (DropsV)[n][p]++;
        //cout <<"lost,"<< n<<",id="<<p<<endl;
    }
}




void Statistic::setLost(simtime_t time) {
    if (time > INI and collect)
        drops++;
}

/*void Statistic::setRouting(int n, int r, double p) {
}*/

void Statistic::setLambda(double l) {
    lambdaMax = l;
}

void Statistic::setGeneration(int genType) {
    genT = genType;
}

void Statistic::setFolder(string folder) {
    folderName = folder;
}



void Statistic::setNumTx(int n) {
    numTx = n;
}

void Statistic::setNumNodes(int n) {
    numNodes = n;
}

void Statistic::setRoutingParaam(double r) {
    routingP = r;
}


void Statistic::printStats() {
    string genString;
    switch (genT) {
        case 0: // Poisson
            genString = "M"; //"Poisson";
            break;
        case 1: // Deterministic
            genString = "D"; //"Deterministic";
            break;
        case 2: // Uniform
            genString = "U"; //"Uniform";
            break;
        case 3: // Binomial
            genString = "B"; //"Binomial";
            break;
        default:
            break;
    }



    vector<double> features;
//    getTrafficInfo();

        //

    // utit
    //int steps = (SIMTIME/1000)+50;
    for (int i = 0; i < numTx; i++) {
       for (int j = 0; j < numTx; j++) {
           long double d = 0;
           int numPackets = (SendP)[i][j];
           int lostPackets = (DropsV)[i][j];
           //cout<<" src="<<i<<" ,dst="<<j<<",num="<<numPackets<<", lost="<<lostPackets<<endl;
           if (lostPackets==0){
               features.push_back(0);

           }else{

               d = (double)lostPackets/(numPackets + lostPackets);
               //cout<<d<<" src="<<i<<" ,dst="<<j<<endl;
               features.push_back(d);
           }
       }
    }

    //features.push_back(drops/steps);

    // Reset
    drops = 0;
    Traffic.clear();
    Delay.clear();


    // Print file
    ofstream myfile;
    //char * filename = new char[80];
    string filename;

    //sprintf(filename, "dataOU_%d_%d_%d_%d_%s.txt",numNodes, numTx, (int) lambdaMax, (int) SIMTIME/1000, genString.c_str());


    // Instant
    filename = folderName + "/PacketLossRate.txt";
//    sprintf(filename, folderName + "/Delay.txt");
    myfile.open (filename, ios::out | ios::trunc );
    for (unsigned int i = 0; i < features.size(); i++ ) {
        double d = features[i];
        myfile  << d << ",";
    }
    myfile << endl;
    myfile.close();


}
