#ifndef polar_listener_cpp
#define polar_listener_cpp

void polar_listener_Create(POLAR_MIXER *Mixer, const char UID[MAX_STRING_LENGTH])
{
    Mixer->Listener->UID = Hash(UID);
    Mixer->Listener->Position = {};
}


//-----------------------------------------------------------------------------
// Purpose: Input handler for fading in volume over time.
// Input  : Float volume fade in time 0 - 100 seconds
//-----------------------------------------------------------------------------
void polar_listener_DistanceFromListener(POLAR_LISTENER *Listener, POLAR_SOURCE_STATE &State, f32 NoiseFloor)
{
    if(State.RolloffDirty)
    {
        State.RolloffFactor = NoiseFloor / powf((State.MaxDistance - State.MinDistance), State.Rolloff);
        State.RolloffDirty = false;
    }

    VECTOR4D Posi = TransformToListener(State.Position, Listener->Position);
    
    State.DistanceFromListener = VectorLength(Posi);
    
    if(State.DistanceFromListener > State.MinDistance)
    {
        State.DistanceAttenuation = (State.RolloffFactor * powf((State.DistanceFromListener - State.MinDistance), State.Rolloff));
    }

    else
    {
        State.DistanceAttenuation = 0.0f;
    }
}

#endif