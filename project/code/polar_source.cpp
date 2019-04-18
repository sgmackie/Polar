


// void polar_source_PlayWAV(ENTITY_SOURCES *Entity, SYS_PLAY *PlaySystem, SYS_WAV *WAVSystem, ID_SOURCE Source, f64 Duration)
// {
//     size_t Index = Entity->RetrieveIndex(Source);
//     ID_SOURCE ID = Entity->IDs[Index];
//     PlaySystem->Start(Entity, ID, Entity->WAVs[Index].SampleRate, Duration, true);
//     WAVSystem->Add(ID);
// }

// void polar_source_Stop(ENTITY_SOURCES *Entity, SYS_PLAY *System, ID_SOURCE Source)
// {
//     size_t Index = Entity->RetrieveIndex(Source);
//     ID_SOURCE ID = Entity->IDs[Index];
//     System->Remove(ID);
//     Entity->Flags[Index] &= ENTITY_SOURCES::PLAYBACK;
// }