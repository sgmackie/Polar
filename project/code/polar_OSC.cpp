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



//!No actual listener search (because there's only one on the master mixer) - needs to be added if there's multiple mixers in the future
void polar_OSC_ProcessListenerEvents(POLAR_MIXER *Mixer, oscpkt::Message *Message, const char FullAddress[MAX_STRING_LENGTH], char Source[MAX_STRING_LENGTH], const char Event[MAX_STRING_LENGTH])
{
    if(Message->match(FullAddress))
    {
        switch(Hash(Event))
        {
            case VECTOR:
            {
                VECTOR4D Vector = {};
                Message->match(FullAddress).popFloat(Vector.X).popFloat(Vector.Y).popFloat(Vector.Z).isOkNoMoreArgs(); 
#if OSC_LOG
                printf("OSC: %s: X: %f Y: %f Z:%f\t[%s]\n", Source, Vector.X, Vector.Y, Vector.Z, FullAddress);
#endif                
                Mixer->Listener->Position = Vector;
                break;
            }

            case ROTATION:
            {
                ROTATION3D Rotation = {};
                Message->match(FullAddress).popFloat(Rotation.Roll).popFloat(Rotation.Pitch).popFloat(Rotation.Yaw).isOkNoMoreArgs(); 
#if OSC_LOG
                printf("OSC: %s: Roll: %f Pitch: %f Yaw:%f\t[%s]\n", Source, Rotation.Roll, Rotation.Pitch, Rotation.Yaw, FullAddress);
#endif                
                Mixer->Listener->Rotation = Rotation;
                break;
            }

            default:
            {
                printf("OSC: Invalid event parameters for %s!\t[%s]\n", Source, FullAddress);
                break;
            }
        }
    }    
}


void polar_OSC_ProcessSourceEvents(POLAR_MIXER *Mixer, f64 GlobalTime, oscpkt::Message *Message, const char FullAddress[MAX_STRING_LENGTH], char Source[MAX_STRING_LENGTH], const char Event[MAX_STRING_LENGTH])
{
    if(Message->match(FullAddress))
    {
        switch(Hash(Event))
        {
            case FADE:
            {
                f32 Amplitude = 0;
                f32 Duration = 0;
                Message->match(FullAddress).popFloat(Amplitude).popFloat(Duration).isOkNoMoreArgs();
                polar_source_Fade(Mixer, Hash(Source), GlobalTime, Amplitude, Duration);
#if OSC_LOG
                printf("OSC: %s: Amplitude: %f\t[%s]\n", Source, Amplitude, FullAddress);
#endif
                break;
            }

            case PLAY:
            {
                i32 Duration = 0;
                f32 Amplitude = 0;
                Message->match(FullAddress).popInt32(Duration).popFloat(Amplitude).isOkNoMoreArgs();
                f32 StackPositions[MAX_CHANNELS] = {0.0};
                polar_source_Play(Mixer, Hash(Source), Duration, StackPositions, FX_DRY, EN_NONE, AMP(Amplitude));
#if OSC_LOG
                printf("OSC: %s: Duration: %d Amplitude: %f\t[%s]\n", Source, Duration, Amplitude, FullAddress);
#endif
                break;
            }

            case VECTOR:
            {
                VECTOR4D Vector = {};
                Message->match(FullAddress).popFloat(Vector.X).popFloat(Vector.Y).popFloat(Vector.Z).isOkNoMoreArgs(); 
#if OSC_LOG
                printf("OSC: %s: X: %f Y: %f Z:%f\t[%s]\n", Source, Vector.X, Vector.Y, Vector.Z, FullAddress);
#endif                
                polar_source_Position(Mixer, Hash(Source), Vector);
                
                break;
            }

            case MATRIX:
            {
                MATRIX_4x4 Matrix = {};
                            
                Message->match(FullAddress).popFloat(Matrix.A1).popFloat(Matrix.A2).popFloat(Matrix.A3).popFloat(Matrix.B1).popFloat(Matrix.B2).popFloat(Matrix.B3).popFloat(Matrix.C1).popFloat(Matrix.C2).popFloat(Matrix.C3).isOkNoMoreArgs();
#if OSC_LOG                            
                printf("OSC: %s: X: %f Y: %f Z:%f\t[%s]\n", Source, Matrix.A1, Matrix.A2, Matrix.A3, FullAddress);
                printf("OSC: %s: X: %f Y: %f Z:%f\t[%s]\n", Source, Matrix.B1, Matrix.B2, Matrix.B3, FullAddress);
                printf("OSC: %s: X: %f Y: %f Z:%f\t[%s]\n", Source, Matrix.C1, Matrix.C2, Matrix.C3, FullAddress);
#endif          
                break;  
            }

            default:
            {
                printf("OSC: Invalid event parameters for %s!\t[%s]\n", Source, FullAddress);
                break;
            }
        }
    }

    else
    {
        printf("OSC: Unhandled message!\n");
    }
}


//-----------------------------------------------------------------------------
// Purpose:
// Input:
//-----------------------------------------------------------------------------
void polar_OSC_UpdateMessages(POLAR_MIXER *Mixer, f64 GlobalTime, UdpSocket &Socket, u32 TimeoutInMS)
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
                switch(Hash(First3Characters))
                {
                    case LN_: 
                    {
                        polar_OSC_ProcessListenerEvents(Mixer, Message, FullAddress, Source, Event);
                        break;
                    }

                    case SO_: 
                    {
                        polar_OSC_ProcessSourceEvents(Mixer, GlobalTime, Message, FullAddress, Source, Event);
                        break;
                    }

                    default:
                    {
                        printf("OSC: Invalid source %s!\t[%s]\n", Source, FullAddress);
                        break;
                    }
                }
            }
        }
    } 
}  

#endif