//TODO: Wishlist:
//! Look up Somberg's state machine 
//TODO: Add option to change WAVEFORMATEX
//TODO: Add option to record to wav
//TODO: Add ability to load and play wav
//TODO: Create audio object structure that's a union of an OSCILLATOR or audio file
//TODO: Finish removing C++ classes and std:: calls
//TODO: More logging (log to text files)

#define POLAR_ASSERT(Expression) if(!(Expression)) {*(int *)0 = 0;}
#define POLAR_MAX_OBJECTS 5

//CRT
#include <stdlib.h>
#include <Windows.h>

//Type defines
#include "../../../misc/includes/win32_types.h"

//Debug
#include "../../../library/debug/debug_macros.h"

//Includes
//Libraries
#include "../../../library/dsp/dsp_wave.h"
#include "../../../library/math/math_vector.h"

//Audio frameworks
#include "../external/win32/wasapi.h"

//Unity build
#include "polar_render.cpp"
#include "polar_object.cpp"
#include "polar_WASAPI.cpp"
#include "polar_log.cpp"


//TODO: Pass array of POLAR_OBJECTs and update individual elements (currently breaking at atomic value?)
void polar_UpdateAndRender(bool &RunningState, KEY_INPUT &Input, POLAR_OBJECT &Object)
{
    //Call for current keyboard input
    KEY KeyCurrent = KeyPress(Input);

    switch(KeyCurrent)
    {
        case Up:
        {            
            //TODO: Interpolate not working over the delta time?
            Object.StreamHandle->AmplitudeCurrent = vector_InterpolateLinear((Object.StreamHandle->AmplitudeCurrent + 0.05f), Object.StreamHandle->AmplitudeCurrent, 0.01f);
            break;
        }
        case Down:
        {
            Object.StreamHandle->AmplitudeCurrent = vector_InterpolateLinear((Object.StreamHandle->AmplitudeCurrent - 0.05f), Object.StreamHandle->AmplitudeCurrent, 0.01f);
            break;
        }
        case Right:
        {
            Object.WaveOscillator->FrequencyCurrent = vector_InterpolateLinear((Object.WaveOscillator->FrequencyCurrent + 10.0f), Object.WaveOscillator->FrequencyCurrent, 5.0f);
            break;
        }
        case Left:
        {
            Object.WaveOscillator->FrequencyCurrent = vector_InterpolateLinear((Object.WaveOscillator->FrequencyCurrent - 10.0f), Object.WaveOscillator->FrequencyCurrent, 5.0f);
            break;
        }
        case Q:
        {
            RunningState = false;
            break;
        }
    }

    //Print states when keyboard input is pressed
    if(KeyCurrent != None)
    {
        polar_log_PrintObjectState(Object);
    }
}

int main(int argc, char *argv[])
{
    //Console input
    SetConsoleTitle("Polar v0.2");
    KEY_INPUT ConsoleInput = {};
    ConsoleInput.WindowsConsole = GetStdHandle(STD_INPUT_HANDLE); //Get windows terminal handle
    
    //Start WASAPI
    polar_WASAPI_Start(0);

    //Create output stream
    RENDER_STREAM *OutputStream = polar_WASAPI_CreateStream(0.25);

    //Create audio objects
    //TODO: Add file reading ("https://social.msdn.microsoft.com/forums/windowsdesktop/en-US/de39afdb-d2fe-40df-98af-12ae446dcf69/basic-wasapi-rendering-audio-question-newbie-stuff") ("https://gist.github.com/fukuroder/7823555") ("https://docs.microsoft.com/en-us/windows/desktop/CoreAudio/rendering-a-stream")
    POLAR_OBJECT_ARRAY ObjectArray = polar_object_CreateObjectArray(POLAR_MAX_OBJECTS);
    POLAR_OBJECT Object01 = polar_object_CreateObject(1, "Wave Oscillator", PLR_OSC);
    
    //TODO: Only one object per-stream; how to mix objects?
    polar_object_SubmitObject(ObjectArray, Object01, OutputStream, PLR_SINE); //Assigns audio object to a rendering stream

    //Print info
    polar_log_PrintAudioFormat(*OutputStream->getAudioFormat());
    polar_log_PrintObjectState(*ObjectArray.Objects[0]);

    //Create rendering thread
    //TODO: Pull out RENDER_STREAM/THREAD into POLAR generic structs so they don't depend on the WASAPI integration to pass audio objects to
    RENDER_THREAD OutputThread(*OutputStream);
    OutputThread.StartThread();

    //Game loop
    bool GlobalRunning = true;
    do
    {
        polar_UpdateAndRender(GlobalRunning, ConsoleInput, *ObjectArray.Objects[0]);
    } while(GlobalRunning);

    //Stop rendering thread
    OutputThread.StopThread();

    //Free allocated structs and data
    polar_object_DestroyObject(*ObjectArray.Objects[0]);
    polar_WASAPI_DestroyStream(OutputStream);

    //Stop WASAPI
    polar_WASAPI_Stop();
}