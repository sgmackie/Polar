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

void polar_OSC_UpdateMessages(POLAR_MIXER *Mixer, UdpSocket &Socket, u32 TimeoutInMS)
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
                const char *Address = 0;
                const char *Event = 0;
                const char *Prefix = 0;
                char Source[MAX_STRING_LENGTH] = {};
                const char *FullAddress = Message->address.c_str();
                const char *Temp = FullAddress;
                Address = strchr(Temp, '/');

                //Slice address into a source ID and an event type
                for(u8 i = 0; i < 2; ++i)
                {
                    if(i == 0)
                    {
                        Prefix = Address+1;

                        strcpy(Source, Prefix);
                        char *Slice = strrchr(Source, '/');  

                        if(Slice) 
                        {
                            *(Slice) = 0;
                        }
                    }
                    
                    if(i == 1)
                    {
                        Event = Address+1;
                    }
                    
                    Address = strchr(Address + 1, '/');
                }

                //Execute events
                if(Message->match(FullAddress))
                {
                    switch(Hash(Event))
                    {
                        case AMPLITUDE:
                        {
                            f32 Amplitude = 0;
                            Message->match(FullAddress).popFloat(Amplitude).isOkNoMoreArgs();
                            polar_source_UpdateAmplitude(Mixer, Source, 0.1, Amplitude);
                            
                            break;
                        }

                        default:
                        {
                            printf("OSC: Invalid event parameters for %s!\t[%s]\n", Source, FullAddress);
                        }
                    }
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