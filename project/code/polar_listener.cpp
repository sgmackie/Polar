#ifndef polar_listener_cpp
#define polar_listener_cpp

void polar_listener_Create(POLAR_MIXER *Mixer, const char UID[MAX_STRING_LENGTH])
{
    Mixer->Listener->UID = Hash(UID);
    Mixer->Listener->Position = {};
}



f32 VectorLength(VECTOR4D A)
{
    f32 Result = 0;
    f32 X = A.X * A.X;
    f32 Y = A.Y * A.Y;
    f32 Z = A.Z * A.Z;
    Result = sqrt((X + Y + Z));
    return Result;
}

VECTOR4D VectorSub(VECTOR4D A, VECTOR4D B)
{
    VECTOR4D Result = {};

    Result.X = A.X - B.X;
    Result.Y = A.Y - B.Y;
    Result.Z = A.Z - B.Z;
    Result.W = A.W - B.W;

    return Result;
}


void MatrixInverse(MATRIX_4x4 &Input, MATRIX_4x4 &Output)
{
    Output.A1 = Input.A1;
    Output.B2 = Input.B2;
    Output.C3 = Input.C3;
    Output.D4 = Input.D4;

    f32 Temp1 = Input.A2;
    f32 Temp2 = Input.B1;
    Output.A2 = Temp2;
    Output.B1 = Temp1;

    Temp1 = Input.A3;
    Temp2 = Input.C1;
    Output.A3 = Temp2;
    Output.C1 = Temp1;

    f32 Temp3 = Input.C2;
    Temp2 = Input.B3;
    Output.C2 = Temp2;
    Output.B3 = Temp3;

    Temp1 = -(Output.A1 * Input.D1 +
            Output.B1 * Input.D2 +
            Output.C1 * Input.D3);

    Temp2 = -(Output.A2 * Input.D1 +
            Output.B2 * Input.D2 +
            Output.C2 * Input.D3);

    Temp3 = -(Output.A3 * Input.D1 +
            Output.B3 * Input.D2 +
            Output.C3 * Input.D3);

    Output.D1 = Temp1;
    Output.D2 = Temp2;
    Output.D3 = Temp3;

    Output.A4 = 0.0f;
    Output.B4 = 0.0f;
    Output.C4 = 0.0f;
}

VECTOR4D ApplyMatrix(VECTOR4D Vector, MATRIX_4x4 Matrix)
{
    VECTOR4D Result = {};

    Result.X = (Vector.X * Matrix.A1) +
                (Vector.Y * Matrix.B1) +
                (Vector.Z * Matrix.C1) +
                (Vector.W * Matrix.D1);
    
    Result.Y = (Vector.X * Matrix.A2) +
                (Vector.Y * Matrix.B2) +
                (Vector.Z * Matrix.C2) +
                (Vector.W * Matrix.D2);
    
    Result.Z = (Vector.X * Matrix.A3) +
                (Vector.Y * Matrix.B3) +
                (Vector.Z * Matrix.C3) +
                (Vector.W * Matrix.D3);
    
    Result.W = (Vector.X * Matrix.A4) +
                (Vector.Y * Matrix.B4) +
                (Vector.Z * Matrix.C4) +
                (Vector.W * Matrix.D4);

    return Result;
}

VECTOR4D TransformToListener(VECTOR4D Source, VECTOR4D Listener)
{
    MATRIX_4x4 InvListener = {};
    InvListener.A2 = -1.0f;
    InvListener.B1 = 1.0f;
    InvListener.C3 = 1.0f;
    InvListener.D4 = 1.0f;

    InvListener.D1 = 0.0f;
    InvListener.D2 = 0.0f;
    InvListener.D3 = 0.0f;

    MatrixInverse(InvListener, InvListener);

    VECTOR4D PosInListenerSpace = VectorSub(Source, Listener);
    VECTOR4D RelativePostion = ApplyMatrix(PosInListenerSpace, InvListener);

    // printf("X: %f\tY: %f\tZ: %f\tW: %f\n", RelativePostion.X, RelativePostion.Y, RelativePostion.Z, RelativePostion.W);

    

    return RelativePostion;
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