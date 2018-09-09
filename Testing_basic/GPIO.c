#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

#define ADDR_GPIO0   0x41200000
#define ADDR_GPIO1   0x41210000
#define ADDR_GPIO2   0x41220000

#define GPIO_TRI_OFFSET 0x4
#define GPIO_CHAN_OFFSET 0x8

#define MAP_SIZE    0x10000
#define MAP_MASK    (MAP_SIZE-1)

void *mapped_base0, *mapped_dev_base0;
void *mapped_base1, *mapped_dev_base1;
void *mapped_base2, *mapped_dev_base2;

off_t gpio_base0 = ADDR_GPIO0;
off_t gpio_base1 = ADDR_GPIO1;
off_t gpio_base2 = ADDR_GPIO2;

void set_GPIO_dir(unsigned int base_addr, unsigned int channel, unsigned int mask)
{
    *(volatile unsigned int *) (base_addr + (channel-1)*GPIO_CHAN_OFFSET + GPIO_TRI_OFFSET) = mask;
}

void set_GPIO_val(unsigned int base_addr, unsigned int channel, unsigned int value)
{
    *(volatile unsigned int *) (base_addr + (channel-1)*GPIO_CHAN_OFFSET) = value;
}

unsigned int get_GPIO_val(unsigned int base_addr, unsigned int channel)
{
    return *(volatile unsigned int *) (base_addr + (channel-1)*GPIO_CHAN_OFFSET);
}

void set_all_write(void)
{
    set_GPIO_dir((unsigned int) mapped_base0, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base0, 2, 0);
    set_GPIO_dir((unsigned int) mapped_base1, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base1, 2, 0);
    set_GPIO_dir((unsigned int) mapped_base2, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base2, 2, 0);
}

void set_all_read(void)
{
    set_GPIO_dir((unsigned int) mapped_base0, 1, 0xFF);
    set_GPIO_dir((unsigned int) mapped_base0, 2, 0xFF);
    set_GPIO_dir((unsigned int) mapped_base1, 1, 0xFF);
    set_GPIO_dir((unsigned int) mapped_base1, 2, 0xFF);
    set_GPIO_dir((unsigned int) mapped_base2, 1, 0xFF);
    set_GPIO_dir((unsigned int) mapped_base2, 2, 0xFF);
}

void set_all_0(void)
{
    set_GPIO_val((unsigned int) mapped_base0, 1, 0);
    set_GPIO_val((unsigned int) mapped_base0, 2, 0);
    set_GPIO_val((unsigned int) mapped_base1, 1, 0);
    set_GPIO_val((unsigned int) mapped_base1, 2, 0);
    set_GPIO_val((unsigned int) mapped_base2, 1, 0);
    set_GPIO_val((unsigned int) mapped_base2, 2, 0);
}

void set_all_1(void)
{
    set_GPIO_val((unsigned int) mapped_base0, 1, 0xFF);
    set_GPIO_val((unsigned int) mapped_base0, 2, 0xFF);
    set_GPIO_val((unsigned int) mapped_base1, 1, 0xFF);
    set_GPIO_val((unsigned int) mapped_base1, 2, 0xFF);
    set_GPIO_val((unsigned int) mapped_base2, 1, 0xFF);
    set_GPIO_val((unsigned int) mapped_base2, 2, 0xFF);
}

/* **********************************
 *              SRAM 1              *
 * CLK             : 33_P
 * A[8:0]          : 
 * DIN_NN[17:0]    : 
 * DIN_NORMAL[7:0] : 
 * ANN             : 
 * DOUT_SEL[3:0]   : 
 ************************************
 * DOUT_NN[9:0]    : 
 ************************************/

/* **********************************
 *              SRAM 2              *
 * CLK             : FMC6[7]                    33_P
 * A[8:0]          : FMC5[7:0] + FMC4[7]        29_P - 25_P
 * DIN_NN[8:0]     : FMC4[6:0] + FMC3[7:6]      25_N - 21_N
 * DIN_NORMAL[7:0] : FMC2[7:0]                  17_P - 14_N
 * DOUT_SEL[5:0]   : FMC3[5:0]                  20_P - 18_N
 * K[4:0]          : NO
 ************************************
 * DOUT_NN[5:0]    : FMC6[5:0]                  32_P - 30_N
 ************************************/
 
 /* **********************************
 *              RRAM               *
 * CLK             : FMC6[7]                    33_P
 * NN_EN[8:0]	   : FMC5[7:0] + FMC4[7]		29_P - 25_P
 * AY[6],AY[3:0]   : FMC3[4:0]					20_N - 18_N
 ************************************
 * DOUT_NN[3:0]    : FMC6[3:0]                  31_P - 30_N
 ************************************/
 
 
void SRAM2_INIT(void)
{
    set_GPIO_dir((unsigned int) mapped_base0, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base0, 2, 0);
    set_GPIO_dir((unsigned int) mapped_base1, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base1, 2, 0);
    set_GPIO_dir((unsigned int) mapped_base2, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base2, 2, 0x7F);
}

void SRAM2_set_A(int value)
{
    unsigned int tmp;
    tmp = (value >> 1) & 0xFF;
    set_GPIO_val((unsigned int) mapped_base2, 1, tmp);
    tmp = get_GPIO_val((unsigned int) mapped_base1, 2);
    tmp = ((value & 0x1) << 7) | (tmp & 0b01111111);
    set_GPIO_val((unsigned int) mapped_base1, 2, tmp);
}

void SRAM2_set_DIN_NN(int value)
{
    unsigned int tmp;
    tmp = get_GPIO_val((unsigned int) mapped_base1, 2);
    tmp = ((value >> 2) & 0x7F) | (tmp & 0b10000000);
    set_GPIO_val((unsigned int) mapped_base1, 2, tmp);
    tmp = get_GPIO_val((unsigned int) mapped_base1, 1);
    tmp = ((value & 0x3) << 6) | (tmp & 0b00111111);
    set_GPIO_val((unsigned int) mapped_base1, 1, tmp);
}

void SRAM2_set_DIN_NORMAL(int value)
{
    set_GPIO_val((unsigned int) mapped_base0, 2, value);  //17_P - 14_N
}

void SRAM2_set_DOUT_SEL(int value)
{
    unsigned int tmp;
    tmp = get_GPIO_val((unsigned int) mapped_base1, 1);
    tmp = (value & 0x3F) | (tmp & 0b11000000);
    set_GPIO_val((unsigned int) mapped_base1, 1, tmp);
}

unsigned int SRAM2_get_DOUT_NN(void)
{
    //return get_GPIO_val((unsigned int) mapped_base2, 2) & 0b111111;
    return get_GPIO_val((unsigned int) mapped_base2, 2) & 0b111;
}

void RRAM_INIT(void)
{
    set_GPIO_dir((unsigned int) mapped_base0, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base0, 2, 0);
    set_GPIO_dir((unsigned int) mapped_base1, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base1, 2, 0);
    set_GPIO_dir((unsigned int) mapped_base2, 1, 0);
    set_GPIO_dir((unsigned int) mapped_base2, 2, 0x7F);
}

void RRAM_set_NNEN(int value)
{
    unsigned int tmp;
    tmp = (value >> 1) & 0xFF;
    set_GPIO_val((unsigned int) mapped_base2, 1, tmp);
    tmp = get_GPIO_val((unsigned int) mapped_base1, 2);
    tmp = ((value & 0x1) << 7) | (tmp & 0b01111111);
    set_GPIO_val((unsigned int) mapped_base1, 2, tmp);
}

void RRAM_set_AY(int value)
{
    unsigned int tmp;
    tmp = get_GPIO_val((unsigned int) mapped_base1, 1);
    tmp = (value & 0x1F) | (tmp & 0b11100000);
    set_GPIO_val((unsigned int) mapped_base1, 1, tmp);
}

unsigned int RRAM_get_DOUT_NN(void)
{
    return get_GPIO_val((unsigned int) mapped_base2, 2) & 0b1111;
}

void simple_delay (int simple_delay)
{
        volatile int i = 0;
        for (i = 0; i < simple_delay; i++);
}

int popcnt(int value)					//有幾個1
{
	int num_one=0;
	for(int i=0;i<9;i++)
			num_one+=(value>>i)&1;
	return num_one;
}

int golden_1bitweight(int ay, int nnen)		//ay 5bit //weight 1bit
{
	int nnen_one_num = popcnt(nnen);
	int sum = 0;

	if(nnen_one_num == 9){
		if(ay == 15)
			sum = -1;
		else if(ay == 31)
			sum = 1;
		else{
			if(ay >= 16){
			sum = nnen_one_num - (ay-16);
			if(sum > 7)
				sum=7;
			if(sum < 0)
				sum = 0;
			}
			else{
				sum = (-nnen_one_num) + ay;
				if(sum < -7)
					sum = -7;
				if(sum > 0)
					sum = 0;
			}
		}
	}
	else{
		if(ay >= 16){
			sum = nnen_one_num - (ay-16);
			if(sum > 7)
				sum=7;
			if(sum < 0)
				sum = 0;
		}
		else{
			sum = (-nnen_one_num) + ay;
			if(sum < -7)
				sum = -7;
			if(sum > 0)
				sum = 0;
		}
	}

	return sum;
}

int set_ay_dependon_macv(int MAC, int nnen){
    int ay;
    int nnen_one_num = popcnt(nnen);
    if(nnen_one_num == 9){
        if(MAC >= nnen_one_num)
            ay = 16;
        else if(MAC <= -nnen_one_num)
            ay = 0;
        else if(MAC > 0 && MAC < nnen_one_num)
            ay = 16 + (nnen_one_num - MAC);
        else if(MAC <= 0 && MAC > -nnen_one_num)
            ay = MAC + nnen_one_num;
        else if(MAC == 1)
            ay = 15;
        else
            ay = 31;
    }
    else{
        if(MAC >= nnen_one_num)
            ay = 16;
        else if(MAC <= -nnen_one_num)
            ay = 0;
        else if(MAC > 0 && MAC < nnen_one_num)
            ay = 16 + (nnen_one_num - MAC);
        else if(MAC <= 0 && MAC > -nnen_one_num)
            ay = MAC + nnen_one_num;
        else
            ay = 14;
    }
    return ay;
}

int set_nnen(int nnref){
	int nnen = 1;
	for(int i = 0; i < nnref; i++)
		nnen = nnen * 2;
	nnen = nnen - 1;
	nnen = nnen << (9-nnref);
	return nnen;
}

int rand_ay(void){
    int ay = 0;
    int ay_6 = 0;
    ay_6 = rand()%2;
    ay = rand()%10;
    if(ay == 9)
        ay = 15;
    if(ay_6 == 1)
        ay = ay + 16;
    return ay;
}

int SEL_control_DOUT(int SEL){
	int DOUT = 0;
	
	return DOUT;
}

int main(int argc, char *argv[])
{
    int device_pointer;
    device_pointer = open("/dev/mem", O_RDWR | O_SYNC);
    
    if (device_pointer < 0){
        printf("device file open error !\n");
        exit(0);
    }
    printf("/dev/mem opened \n");

    mapped_base0 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, device_pointer, gpio_base0);
    mapped_base1 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, device_pointer, gpio_base1);
    mapped_base2 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, device_pointer, gpio_base2);

    /*
     *  Start the codes
     */
	//SRAM2_INIT();
	RRAM_INIT();
	
	//SRAM2_set_A(0b111111111);
	//SRAM2_set_DIN_NN(0b111111100);
	//SRAM2_set_DOUT_SEL(0b111);

	//printf("result: %x\n", SRAM2_get_DOUT_NN());
    
	int dout_gold =0;
		int err =0;
		srand(0);

		for(int i=0;i<100;i++){
			//nagedge
			int ay = rand_ay();
			//int ay = 16;
			//int nnen =rand()%512;
			int nnen =511;
			int nnref = popcnt(nnen);		//nnen有幾個1, nnref 就設多少
			int dout;
			
			RRAM_set_AY(ay);
			//RRAM_set_NNEN(set_nnen(nnref));
			RRAM_set_NNEN(nnen);
			usleep(1000000);
			dout = RRAM_get_DOUT_NN();

			if((dout & 0b1000) == 8)
				dout = -(dout & 0b111);
			else
				dout = dout & 0b1111;
			
			dout_gold = golden_1bitweight(ay,nnen);
			if(dout != dout_gold){
				printf("ay: %d ,ans: %d, result: %d\n",ay, dout_gold, dout);
				err++;
			}
		}



		while(1){
			RRAM_set_AY(0b10000);
			RRAM_set_NNEN(0b111111111);
			SRAM2_set_A(0b111111111);
		}

		printf("hahaha");
		
	
    return(0);
}
