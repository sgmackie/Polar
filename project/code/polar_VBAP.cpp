


void findLsPairs(size_t **SpeakerPairs, f64 *Directions, size_t *Indexes, size_t Count, bool OmitLargePairs = false, f32 ApertureLimit = 0)
{
    printf("Original\n");
    for(size_t i=0; i<Count; i++)
    {
        printf("%f, ", Directions[i]);
    }
    printf("\n");

    f64 *arr = Directions;
    

    for(size_t i=0; i<Count; i++)
    {
        /* 
         * Place currently selected element array[i]
         * to its correct place.
         */
        for(size_t j=i+1; j<Count; j++)
        {
            /* 
             * Swap if currently selected array element
             * is not at its correct position.
             */
            if(arr[i] > arr[j])
            {
                f32 temp     = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;

                size_t index = Indexes[i];

                Indexes[i] = Indexes[j];
                Indexes[j] = index;
            }
        }

    }

    printf("Sort\n");
    for(size_t i=0; i<Count; i++)
    {
        printf("%f, ", arr[i]);
    }
    printf("\n");


    Indexes[5] = Indexes[0];
    printf("Indexes\n");
    for(size_t i=0; i<Count+1; i++)
    {
        printf("%zu, ", Indexes[i]);
    }
    printf("\n");


    for(int i = 0; i < Count; ++i)
    {
        SpeakerPairs[i][0] = Indexes[i];
        SpeakerPairs[i][1] = Indexes[i+1];
    }

    printf("Pairs\n");
    for(size_t i=0; i<Count; i++)
    {
        printf("%zu | %zu, ", SpeakerPairs[i][0], SpeakerPairs[i][1]);
    }
    printf("\n");

    return;
}


void invertLsMtx2D(f64 **Matrix, size_t **SpeakerPairs, f64 *Directions, f64 *Radians, f64 **Cartesians, size_t Count)
{
    for(int i = 0; i < Count; ++i)
    {
        Radians[i] = (Directions[i] * (PI32 / 180));
    }

    printf("Radians\n");
    for(size_t i=0; i<Count; i++)
    {
        printf("%f, ", Radians[i]);
    }
    printf("\n");



    for(int i = 0; i < Count; ++i)
    {
        Cartesians[i][0] = (1.0 * cos(Radians[i]));
        Cartesians[i][1] = (1.0 * sin(Radians[i]));
    }

    printf("Cartesians\n");
    for(size_t i=0; i<Count; i++)
    {
        printf("%f | %f, ", Cartesians[i][0], Cartesians[i][1]);
    }
    printf("\n");


    f64 Temp[2][2];
    for(int i = 0; i < Count; ++i)
    {
        size_t A = SpeakerPairs[i][0];
        size_t B = SpeakerPairs[i][1];
        
        //Off by 1 error
        A -= 1;
        B -= 1;

        Temp[0][0] = Cartesians[A][0];
        Temp[0][1] = Cartesians[A][1];
        Temp[1][0] = Cartesians[B][0];
        Temp[1][1] = Cartesians[B][1];

        printf("%f | %f, %f | %f\n", Temp[0][0], Temp[0][1], Temp[1][0], Temp[1][1]);  
    }


    return;
}


int test()
{
    //Stage 1: Set speaker directions
    f64 LoudSpeakers[5] = {30, -30, 0, 110, -110};
    f64 Copy[5] = {30, -30, 0, 110, -110};
    size_t Indexes[6] = {1, 2, 3, 4, 5, 0}; //Create one extra element to loop back on
    
    size_t **SpeakerPairs = (size_t **) malloc(sizeof(size_t) * 5);
    for(int i = 0; i < 5; ++i)
    {   
        SpeakerPairs[i] = (size_t *) malloc(sizeof(size_t) * 2);
    }

    //Stage 2: Find speaker pairs
    findLsPairs(SpeakerPairs, Copy, Indexes, 5);
    
    //Stage 3: Find inverse matrices
    f64 Radians[5];

    f64 **Cartesians = (f64 **) malloc(sizeof(f64) * 5);
    for(int i = 0; i < 5; ++i)
    {   
        Cartesians[i] = (f64 *) malloc(sizeof(f64) * 2);
    }

    f64 **Matrix = (f64 **) malloc(sizeof(f64) * 5);
    for(int i = 0; i < 5; ++i)
    {   
        //2D = 2 * 2, 3D = 3 * 3;
        Matrix[i] = (f64 *) malloc(sizeof(f64) * 4);
    }

    invertLsMtx2D(Matrix, SpeakerPairs, LoudSpeakers, Radians, Cartesians, 5);
    
    free(SpeakerPairs);
    return 0;
}