#ifndef polar_cpp
#define polar_cpp

//Iterate through strings A & B and push to string C
void polar_StringConcatenate(char *StringA, size_t StringALength, char *StringB, size_t StringBLength, char *StringC)
{
    for(i32 Index = 0; Index < StringALength; ++Index)
    {
        *StringC++ = *StringA++;
    }

    for(i32 Index = 0; Index < StringBLength; ++Index)
    {
        *StringC++ = *StringB++;
    }

    *StringC++ = 0;
}


i32 polar_StringLengthGet(char *String)
{
    i32 Length = 0;
    
    //Increment through string until null
    while(*String++)
    {
        ++Length;
    }

    return(Length);
}

#endif