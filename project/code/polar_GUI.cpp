#ifndef polar_GUI_cpp
#define polar_GUI_cpp

void polar_GUI_TreeNodeCreate(POLAR_SUBMIX *SubmixIndex)
{
    if(ImGui::TreeNode(SubmixIndex->Name))
    {
        for(u32 ContainerIndex = 0; ContainerIndex < SubmixIndex->Containers.CurrentContainers; ++ContainerIndex)
        {
            if(ImGui::TreeNode(SubmixIndex->Containers.Name[ContainerIndex]))
            {
                u32 SourceSelect = 0;
                char SourceLabel[MAX_STRING_LENGTH];

                for(u32 SourceIndex = 0; SourceIndex < SubmixIndex->Containers.Sources[ContainerIndex].CurrentSources; ++SourceIndex)
                {       
                    sprintf(SourceLabel, SubmixIndex->Containers.Sources[ContainerIndex].Name[SourceIndex]);
                    if(ImGui::Selectable(SourceLabel, SourceSelect == SourceIndex))
                    {
                        SourceSelect = SourceIndex;
                    }
                }

                ImGui::TreePop();
            }
        }
    }
}


void polar_GUI_Construct(POLAR_MIXER *Mixer)
{
    ImGui::Begin("Master");
    ImGui::SliderFloat("Master", &Mixer->Amplitude, 0.0f, 1.0f, "%.4f", 1.0f);
    ImGui::End();


    ImGui::Begin("Sources");

    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        if(ImGui::TreeNode(SubmixIndex->Name))
        {
            for(u32 ContainerIndex = 0; ContainerIndex < SubmixIndex->Containers.CurrentContainers; ++ContainerIndex)
            {
                if(ImGui::TreeNode(SubmixIndex->Containers.Name[ContainerIndex]))
                {
                    u32 SourceSelect = 0;
                    char SourceLabel[MAX_STRING_LENGTH];

                    for(u32 SourceIndex = 0; SourceIndex < SubmixIndex->Containers.Sources[ContainerIndex].CurrentSources; ++SourceIndex)
                    {       
                        sprintf(SourceLabel, SubmixIndex->Containers.Sources[ContainerIndex].Name[SourceIndex]);
                        if(ImGui::Selectable(SourceLabel, SourceSelect == SourceIndex))
                        {
                            SourceSelect = SourceIndex;
                        }
                    }

                    ImGui::TreePop();
                }
            }

            for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
            {
                polar_GUI_TreeNodeCreate(ChildSubmixIndex);
            }

            ImGui::TreePop();
        }
    }
    
    ImGui::End();


    ImGui::Begin("Log");
    ImGui::End();


}




#endif