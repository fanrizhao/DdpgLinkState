package networksimulator;

network NetworkAll
{
    parameters:
        int numNodes = 4;
        int numTx = 4;

    types:
        channel Channel extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 9 kbps;
        }

        submodules:
            node[numNodes]: Server {
                id = index;
                numTx = numTx;
                numNodes = numNodes;
                //gates:
                //    port[3];
            }

    connections:
        node[0].port[0] <--> Channel <--> node[2].port[0];
        node[1].port[0] <--> Channel <--> node[2].port[1];
        node[2].port[2] <--> Channel <--> node[3].port[2];
}
