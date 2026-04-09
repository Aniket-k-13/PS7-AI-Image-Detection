#include <stdio.h>
#include <string.h>

char M[100][4];
char IR[4];
char R[4];
char buffer[40];

int IC = 0;
int C = 0;
int SI = 0;

void START_EXECUTION();
void EXECUTE_USER_PROGRAM();
void MOS();
void READ();
void WRITE();
void TERMINATE();

int main()
{
    // Initialize memory with spaces
    memset(M, ' ', sizeof(M));

    // Load program (like your board example)
    strcpy(M[0], "GD20");
    strcpy(M[1], "PD20");
    strcpy(M[2], "H   ");

    printf("Enter Data: ");
    
    START_EXECUTION();

    return 0;
}

void START_EXECUTION()
{
    IC = 0;
    EXECUTE_USER_PROGRAM();
}

void EXECUTE_USER_PROGRAM()
{
    while (1)
    {
        // FETCH
        for (int i = 0; i < 4; i++)
            IR[i] = M[IC][i];

        IC++;

        if (IR[0]=='G' && IR[1]=='D')
        {
            SI = 1;
            MOS();
        }

     
        else if (IR[0]=='P' && IR[1]=='D')
        {
            SI = 2;
            MOS();
        }

       
        else if (IR[0]=='H')
        {
            SI = 3;
            MOS();
            break;
        }

        // LR
        else if (IR[0]=='L' && IR[1]=='R')
        {
            int addr = (IR[2]-'0')*10 + (IR[3]-'0');
            for (int i=0;i<4;i++)
                R[i] = M[addr][i];
        }

      
        else if (IR[0]=='S' && IR[1]=='R')
        {
            int addr = (IR[2]-'0')*10 + (IR[3]-'0');
            for (int i=0;i<4;i++)
                M[addr][i] = R[i];
        }

        
        else if (IR[0]=='C' && IR[1]=='R')
        {
            int addr = (IR[2]-'0')*10 + (IR[3]-'0');
            C = 1;
            for (int i=0;i<4;i++)
            {
                if (R[i] != M[addr][i])
                {
                    C = 0;
                    break;
                }
            }
        }

        
        else if (IR[0]=='B' && IR[1]=='T')
        {
            int addr = (IR[2]-'0')*10 + (IR[3]-'0');
            if (C == 1)
                IC = addr;
        }
    }
}

void MOS()
{
    if (SI == 1)
        READ();
    else if (SI == 2)
        WRITE();
    else if (SI == 3)
        TERMINATE();
}

void READ()
{
    scanf("%s", buffer);

    int k = 0;
    int addr = (IR[2]-'0')*10 + (IR[3]-'0');

    for (int i = addr; i < addr + 10; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (buffer[k] != '\0')
                M[i][j] = buffer[k++];
            else
                M[i][j] = ' ';
        }
    }
}

void WRITE()
{
    int addr = (IR[2]-'0')*10 + (IR[3]-'0');

    printf("\nOutput: ");
    for (int i = addr; i < addr + 10; i++)
    {
        for (int j = 0; j < 4; j++)
            printf("%c", M[i][j]);
    }
    printf("\n");
}

void TERMINATE()
{
    printf("\nProgram Terminated\n");
}