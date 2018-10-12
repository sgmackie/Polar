#ifndef polar_log_h
#define polar_log_h

//Print current state of render OutputThread
void polar_log_PrintParameterStates(RENDER_STREAM &OutputStream, OSCILLATOR &InputOscillator);

//Print current sample rate and bit depth
void polar_log_PrintAudioFormat(AUDIO_FORMAT &audioFormat);

#endif