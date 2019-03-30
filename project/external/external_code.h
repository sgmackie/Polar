#ifndef external_code_h
#define external_code_h

//Hashing function for IDs
#include "xxhash.c"

//Random number generator
#include "pcg_basic.c"

//WAV handling 
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

//OSC messaging
#define OSCPKT_OSTREAM_OUTPUT
#include "oscpkt.hh"
#include "udp.hh"

//IMGUI
#include "imgui/imgui.cpp"
#include "imgui/imgui_widgets.cpp"
#include "imgui/imgui_draw.cpp"
#include "imgui/imgui_demo.cpp"

#endif