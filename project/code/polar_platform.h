#ifndef polar_platform_h
#define polar_platform_h

#include "polar_WASAPI.cpp"
#include "polar_platform.h"

//Create and initialise WASAPI struct
WASAPI_DATA *polar_WASAPI_Create(WASAPI_BUFFER &Buffer);

//Remove WASAPI struct
void polar_WASAPI_Destroy(WASAPI_DATA *WASAPI);


//Get WASAPI buffer and release after filling with specified amount of samples
void polar_WASAPI_PrepareBuffer(WASAPI_DATA *WASAPI, WASAPI_BUFFER &Buffer);


void polar_WASAPI_ReleaseBuffer(WASAPI_DATA *WASAPI, WASAPI_BUFFER &Buffer);


//Update the audio clock's position in the current stream
void polar_WASAPI_UpdateClock(WASAPI_DATA &Interface, WASAPI_CLOCK Clock);


#endif