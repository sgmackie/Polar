#ifndef polar_OSC_cpp
#define polar_OSC_cpp

using namespace oscpkt;

//OSC messages
//Source types
#define LN_                     9201655152285363179U
#define SO_                     8744316438972908U

//Events
#define PLAY                    11120484276852016966U
#define FADE                    7677966677680727406U
#define VECTOR                  12143376858605269818U
#define ROTATION                405295350552126795U
#define MATRIX                  16755126490873392952U

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

void polar_OSC_ProcessSourceEvents(ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, f64 GlobalTime, oscpkt::Message *Message, const char FullAddress[MAX_STRING_LENGTH], char Source[MAX_STRING_LENGTH], const char Event[MAX_STRING_LENGTH])
{
    if(Message->match(FullAddress))
    {
        switch(FastHash(Event))
        {
            case VECTOR:
            {
                CMP_POSITION Vector = {};
                Message->match(FullAddress).popFloat(Vector.X).popFloat(Vector.Y).popFloat(Vector.Z).popFloat(Vector.W).isOkNoMoreArgs();
                // Debug("OSC: X: %f Y: %f Z: %f W: %f\t[%s]", Vector.X, Vector.Y, Vector.Z, Vector.W, FullAddress);

                
				HANDLE_SOURCE Handle = Sources->RetrieveHandle(FastHash(Source));
				
                for(size_t i = 0; i < Voices->Count; ++i)
                {
                    if(FastHash(Source) == Voices->Sources[i].ID)
                    {
                        Voices->Positions[i] = Vector;
                    }
                }

                

                break;
            }
        }
    }

    else
    {
        Warning("OSC: Unhandled message!\n");
    }    
}

void polar_OSC_UpdateMessages(ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, LISTENER *GlobalListener, f64 GlobalTime, UdpSocket &Socket, u32 TimeoutInMS)
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
                char First3Characters[4] = {};
                strncpy(First3Characters, Source, 3);
                First3Characters[3] = 0;
                switch(FastHash(First3Characters))
                {
                    case LN_: 
                    {
                        // polar_OSC_ProcessListenerEvents(Mixer, Message, FullAddress, Source, Event);
                        if(Message->match(FullAddress))
                        {
                            switch(FastHash(Event))
                            {
                                case VECTOR:
                                {
                                    CMP_POSITION Vector = {};
                                    Message->match(FullAddress).popFloat(Vector.X).popFloat(Vector.Y).popFloat(Vector.Z).isOkNoMoreArgs();
                                    GlobalListener->Position.X = Vector.X;
                                    GlobalListener->Position.Y = Vector.Y;
                                    GlobalListener->Position.Z = Vector.Z;
                                    break;
                                }
                                case ROTATION:
                                {
                                    CMP_POSITION Vector = {};
                                    Message->match(FullAddress).popFloat(Vector.X).popFloat(Vector.Y).popFloat(Vector.Z).isOkNoMoreArgs();
                                    GlobalListener->Rotation.X = Vector.X;
                                    GlobalListener->Rotation.Y = Vector.Y;
                                    GlobalListener->Rotation.Z = Vector.Z;
                                    break;
                                }                                
                            }
                        }

                        else
                        {
                            Warning("OSC: Unhandled message!\n");
                        }                         
                        break;
                    }

                    case SO_: 
                    {
                        polar_OSC_ProcessSourceEvents(Sources, Voices, GlobalTime, Message, FullAddress, Source, Event);
                        break;
                    }

                    default:
                    {
                        Warning("OSC: Invalid source %s!\t[%s]\n", Source, FullAddress);
                        break;
                    }
                }
            }
        }
    } 
}  

#endif