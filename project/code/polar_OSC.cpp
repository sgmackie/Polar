#ifndef polar_OSC_cpp
#define polar_OSC_cpp

using namespace oscpkt;


UdpSocket polar_OSC_StartServer(u32 Port)
{    
    UdpSocket Result = {};
    Result.bindTo(Port);
    
    if(!Result.isOk()) 
    {
        printf("OSC: Failed to create UDP socket at port %d!\n", Port);
        Result = {};
        return Result;
    }

    printf("OSC: Listening on port %d\n", Port);

    return Result;
}

i32 polar_OSC_StartEngine(UdpSocket &Socket, u32 TimeoutInSeconds)
{
    if(Socket.isOk())
    {
        PacketReader PacketReader;
        TimeoutInSeconds *= 1000;

        if(Socket.receiveNextPacket(TimeoutInSeconds))
        {
            oscpkt::Message *Message;

            //Get packets
            void *PacketData = Socket.packetData();
            size_t PacketSize = Socket.packetSize();
            PacketReader.init(PacketData, PacketSize);

            const char *FullAddress = "POLAR_START";

            while(PacketReader.isOk() && (Message = PacketReader.popMessage()) != 0) 
            {
                return 1;
            }
        }
    }

    return 0;
}

void polar_OSC_UpdateMessages(f64 GlobalTime, UdpSocket &Socket, u32 TimeoutInMS)
{
    if(Socket.isOk())
    {
        PacketReader PacketReader;

        if(Socket.receiveNextPacket(TimeoutInMS))
        {
            oscpkt::Message *Message;

            //Get packets
            void *PacketData = Socket.packetData();
            size_t PacketSize = Socket.packetSize();
            PacketReader.init(PacketData, PacketSize);

            while(PacketReader.isOk() && (Message = PacketReader.popMessage()) != 0) 
            {
                //Find OSC addresses
                char const *FullAddress = Message->address.c_str();

                printf("%s\n", FullAddress);
            }
        }
    } 
}  

#endif