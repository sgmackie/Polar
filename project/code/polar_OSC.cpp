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

    return Result;
}


void polar_OSC_UpdateMessages(UdpSocket &Socket, u32 TimeoutInMS)
{
    if(Socket.isOk())
    {
        PacketReader PacketReader;

        if(Socket.receiveNextPacket(TimeoutInMS))
        {
            oscpkt::Message *Message;
            f32 Data = 0;

            void *PacketData = Socket.packetData();
            size_t PacketSize = Socket.packetSize();
            
            PacketReader.init(PacketData, PacketSize);

            while(PacketReader.isOk() && (Message = PacketReader.popMessage()) != 0) 
            {
                if(Message->match("/Polar").popFloat(Data).isOkNoMoreArgs())
                {
                    printf("OSC: /Polar\t%f\n", Data);
                }

                else
                {
                    printf("OSC: Unhandled message!\n");
                }
            }
        }
    } 
}

#endif