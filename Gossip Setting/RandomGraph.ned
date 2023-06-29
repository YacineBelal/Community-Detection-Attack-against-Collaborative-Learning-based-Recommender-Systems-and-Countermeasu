simple Server
{
    gates:
        inout sl[];

}
simple Node
{
    gates:
        inout no[943];
        inout nl[1];

}

network RandomGraph
{
    parameters:
        int size = default(943);
        
    submodules: 
        server: Server {
            @display("p=339,24");
        }
         Noeuds[size] : Node;

    connections allowunconnected:
        for i = 0 .. size - 1{ 
        server.sl++ <--> {} <--> Noeuds[i].nl++;
        }
        for i = 0 .. size - 1,for k = i + 1  .. size - 1{
        Noeuds[i].no++ <--> {} <--> Noeuds[k].no++;     
        }
           
}
