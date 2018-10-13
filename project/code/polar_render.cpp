// void polar_render_RenderCallback(OSCILLATOR &Object, BYTE *Buffer, u32 BufferSizeInFrames)
// {
//     //Set the buffer start and end points
//     BYTE *BufferStartPoint = Buffer;
//     BYTE *BufferEndPoint = (Buffer + (BufferSizeInFrames * m_spAudioFormat->getFrameSizeInBytes()));

//     //Loop until the start point reaches the end point
//     while(BufferStartPoint < BufferEndPoint)
//     {
//         // f64 CurrentSample = sampleSource.nextSample(m_spAudioFormat->getSampleRateInHz()) * AmplitudeCurrent;
//         f64 CurrentSample = (dsp_wave_TickSine(&Object) * AmplitudeCurrent);

//         //TODO: Add panning information
//         for(int i = 0; i < m_spAudioFormat->getNumChannels(); ++i)
//         {
//             switch (m_spAudioFormat->getEncoding())
//             {
//                 case AudioEncoding::FLOATING_POINT:
//                 {
//                     SampleTypeConverter::doubleToFloat(CurrentSample, *reinterpret_cast<float *>(BufferStartPoint));
//                     BufferStartPoint += sizeof(float);
//                     break;
//                 }
//                 case AudioEncoding::PCM_8:
//                 {
//                     SampleTypeConverter::doubleTo8BitUnsigned(CurrentSample, *reinterpret_cast<uint8_t *>(BufferStartPoint));
//                     BufferStartPoint += 1;
//                     break;
//                 }
//                 case AudioEncoding::PCM_16:
//                 {
//                     SampleTypeConverter::doubleTo16BitSigned(CurrentSample, *reinterpret_cast<int16_t *>(BufferStartPoint));
//                     BufferStartPoint += 2;
//                     break;
//                 }
//                 case AudioEncoding::PCM_24:
//                 {                    
//                     SampleTypeConverter::doubleTo24BitSigned(CurrentSample, BufferStartPoint);
//                     BufferStartPoint += 3;
//                     break;
//                 }
//                 case AudioEncoding::PCM_24_IN_32:
//                 {
//                     SampleTypeConverter::doubleTo24BitIn32Signed(CurrentSample, BufferStartPoint);
//                     BufferStartPoint += 4;
//                     break;
//                 }
//                 case AudioEncoding::PCM_32:
//                 {
//                     SampleTypeConverter::doubleTo32BitSigned(CurrentSample, *reinterpret_cast<int32_t *>(BufferStartPoint));
//                     BufferStartPoint += 4;
//                     break;
//                 }
//                 default:
//                 {
//                     debug_PrintLine(Console, "WASAPI: Unsupported audio format!");
//                 }
//             }
//         }
//     }
// }
