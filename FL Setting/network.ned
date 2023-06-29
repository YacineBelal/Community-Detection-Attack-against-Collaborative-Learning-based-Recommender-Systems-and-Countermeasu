simple Server
{
    gates:
        inout sl[];

}

simple Node
{
    gates:

        inout nl[];

}

network FLnetwork
{
    submodules:
        server: Server;
         Noeuds[943] : Node;
    connections:
        for i = 0 .. 99{ 
        server.sl++ <--> {} <--> Noeuds[i].nl++;
        }

}
