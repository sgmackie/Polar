//CRT
#include <stdlib.h>
#include <stdio.h>


//"https://soundprogramming.net/programming/alsa-tutorial-1-initialization/"
//libasound2 apt get
#include <alsa/asoundlib.h>



//Polar
//TODO: Get dynamic loading working in Clang
// #include "polar.h"



snd_pcm_t *linux_ALSA_AudioDeviceCreate()
{
    printf("ALSA: Start\n");

    snd_pcm_t *Device;
    Device = {};
    return Device;
}



void linux_ALSA_AudioDeviceDestroy(snd_pcm_t *Device)
{
    snd_pcm_close(Device);
    printf("ALSA: Closed device and unitialised\n");
}




int main(int argc, char* argv[])
{
    
    snd_pcm_t *SoundDevice = linux_ALSA_AudioDeviceCreate();
    linux_ALSA_AudioDeviceDestroy(SoundDevice);


    return 0;
}