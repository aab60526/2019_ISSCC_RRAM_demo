#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_IMG	50

static float**** Conv1_float;

////////////////Conv2blk1_parameter///////////////
static float* conv2blk1_bn_beta;
static float* conv2blk1_bn_gamma;
static float* conv2blk1_bn_mean;
static float* conv2blk1_bn_variance;
static float* conv2blk1_stembn_beta;
static float* conv2blk1_stembn_gamma;
static float* conv2blk1_stembn_mean;
static float* conv2blk1_stembn_variance;
static float**** Conv2blk1_c3x3a_float;
static float**** Conv2blk1_c3x3b_float;
////////////////Conv2blk1_parameter////////////////

////////////////Conv2blk2_parameter///////////////
static float* conv2blk2_bn_beta;
static float* conv2blk2_bn_gamma;
static float* conv2blk2_bn_mean;
static float* conv2blk2_bn_variance;
static float* conv2blk2_stembn_beta;
static float* conv2blk2_stembn_gamma;
static float* conv2blk2_stembn_mean;
static float* conv2blk2_stembn_variance;
static float**** Conv2blk2_c3x3a_float;
static float**** Conv2blk2_c3x3b_float;
////////////////Conv2blk2_parameter////////////////

////////////////Conv2blk3_parameter///////////////
static float* conv2blk3_bn_beta;
static float* conv2blk3_bn_gamma;
static float* conv2blk3_bn_mean;
static float* conv2blk3_bn_variance;
static float* conv2blk3_stembn_beta;
static float* conv2blk3_stembn_gamma;
static float* conv2blk3_stembn_mean;
static float* conv2blk3_stembn_variance;
static float**** Conv2blk3_c3x3a_float;
static float**** Conv2blk3_c3x3b_float;
////////////////Conv2blk3_parameter////////////////

////////////////Conv3blk1_parameter///////////////
static float* conv3blk1_bn_beta;
static float* conv3blk1_bn_gamma;
static float* conv3blk1_bn_mean;
static float* conv3blk1_bn_variance;
static float* conv3blk1_stembn_beta;
static float* conv3blk1_stembn_gamma;
static float* conv3blk1_stembn_mean;
static float* conv3blk1_stembn_variance;
static float**** Conv3blk1_c3x3a_float;
static float**** Conv3blk1_c3x3b_float;
////////////////Conv3blk1_parameter////////////////

////////////////Conv3blk2_parameter///////////////
static float* conv3blk2_bn_beta;
static float* conv3blk2_bn_gamma;
static float* conv3blk2_bn_mean;
static float* conv3blk2_bn_variance;
static float* conv3blk2_stembn_beta;
static float* conv3blk2_stembn_gamma;
static float* conv3blk2_stembn_mean;
static float* conv3blk2_stembn_variance;
static float**** Conv3blk2_c3x3a_float;
static float**** Conv3blk2_c3x3b_float;
////////////////Conv3blk2_parameter////////////////

////////////////Conv3blk3_parameter///////////////
static float* conv3blk3_bn_beta;
static float* conv3blk3_bn_gamma;
static float* conv3blk3_bn_mean;
static float* conv3blk3_bn_variance;
static float* conv3blk3_stembn_beta;
static float* conv3blk3_stembn_gamma;
static float* conv3blk3_stembn_mean;
static float* conv3blk3_stembn_variance;
static float**** Conv3blk3_c3x3a_float;
static float**** Conv3blk3_c3x3b_float;
////////////////Conv3blk3_parameter////////////////

////////////////Conv4blk1_parameter///////////////
static float* conv4blk1_bn_beta;
static float* conv4blk1_bn_gamma;
static float* conv4blk1_bn_mean;
static float* conv4blk1_bn_variance;
static float* conv4blk1_stembn_beta;
static float* conv4blk1_stembn_gamma;
static float* conv4blk1_stembn_mean;
static float* conv4blk1_stembn_variance;
static float**** Conv4blk1_c3x3a_float;
static float**** Conv4blk1_c3x3b_float;
////////////////Conv4blk1_parameter////////////////

////////////////Conv4blk2_parameter///////////////
static float* conv4blk2_bn_beta;
static float* conv4blk2_bn_gamma;
static float* conv4blk2_bn_mean;
static float* conv4blk2_bn_variance;
static float* conv4blk2_stembn_beta;
static float* conv4blk2_stembn_gamma;
static float* conv4blk2_stembn_mean;
static float* conv4blk2_stembn_variance;
static float**** Conv4blk2_c3x3a_float;
static float**** Conv4blk2_c3x3b_float;
////////////////Conv4blk2_parameter////////////////

////////////////Conv4blk3_parameter///////////////
static float* conv4blk3_bn_beta;
static float* conv4blk3_bn_gamma;
static float* conv4blk3_bn_mean;
static float* conv4blk3_bn_variance;
static float* conv4blk3_stembn_beta;
static float* conv4blk3_stembn_gamma;
static float* conv4blk3_stembn_mean;
static float* conv4blk3_stembn_variance;
static float**** Conv4blk3_c3x3a_float;
static float**** Conv4blk3_c3x3b_float;
////////////////Conv4blk3_parameter////////////////

/////////////////fc and lastbn/////////////////////
static float* lastbn_beta;
static float* lastbn_gamma;
static float* lastbn_mean;
static float* lastbn_variance;
static float* fct_b;
static float** fct_W;
/////////////////fc and lastbn/////////////////////

/////////////////input/////////////////////
static float *classes;
static float*** inputs[N_IMG];
static float*** input_mean_0;
static float*** input_mean_1;
static float*** input_mean_2;
/////////////////input/////////////////////
float*** Load_input_float(int input_size, int input_channel,FILE* fn){
    float*** inputt;
    float read_in;
    int i,j,k,l;
    int ii=0;
    inputt = malloc(input_size * sizeof(float**));
    for(i=0;i<input_size;i++){
        inputt[i] = malloc(input_size * sizeof(float*));
        for(j=0;j<input_size;j++){
            inputt[i][j] = malloc(input_channel * sizeof(float));

        }
    }

        for(k=0;k<input_channel;k++){
            for(j=0;j<input_size;j++){
                for(i=0;i<input_size;i++){
                    fscanf(fn,"%f",&read_in);
                    inputt[i][j][k] = read_in;
                    //printf("%hd",read_in);
                    //printf("FFFFFFFFFFFFFFFFFFFFFFF");
                }
            }
           // printf("\n");
        }
    return inputt;
}


static void load_input(void){
    FILE *fp;
	int foo;
	float v;

	if ((fp = fopen("input.txt", "r")) == NULL){
        fprintf(stderr, "Fail opening input.txt\n");
		exit(1);
	}

	for (int i = 0; i < N_IMG; i++){
		inputs[i] = malloc(sizeof(float **) * 32);
		for (int j = 0; j < 32; j++){
			inputs[i][j] = malloc(sizeof(float*) * 32);
			for (int k = 0; k < 32; k++)
				inputs[i][j][k] = malloc(sizeof(float) * 3);
		}
	}

	classes = malloc(sizeof(float ) * N_IMG);
    printf("build successful\n");


	for(int n = 0; n < N_IMG; n++){
		foo = fscanf(fp, "%f\n", &v);
		classes[n] = v;
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 32; j++){
				for(int k = 0; k < 32; k++){
					foo = fscanf(fp, "%f", &v);
					inputs[n][k][j][i] = v;
				}
				foo = fscanf(fp, "\n");
			}
			foo = fscanf(fp, "\n");
		}
	}

	fp = fopen("pp_mean_0.txt","r");
    input_mean_0 = Load_input_float(32,1,fp);
    fp = fopen("pp_mean_1.txt","r");
    input_mean_1 = Load_input_float(32,1,fp);
    fp = fopen("pp_mean_2.txt","r");
    input_mean_2 = Load_input_float(32,1,fp);


	fclose(fp);

	printf("build successful\n");
}

float**** load_CNN_weight_float(int width, int hight, int input_channel, int howmany_filter,FILE* fn){//1 50
    int test_count;
    //int CNN_weight[width][hight][input_channel][howmany_filter];
    float ****CNN_weight;
    float read_in;
    int i,j,k,m,l;
    char test22;
    int change;
    change = howmany_filter;
    howmany_filter = input_channel;
    input_channel = change;

    //FILE *fn;
    //fn = fopen("kernel2.txt","r");
    //FILE *fn;
    //fn = fopen("kernel1.txt","r");
    test_count = 0;
    CNN_weight =  malloc(width * sizeof(float***));
    for(k=0;k<width;k++){
        CNN_weight[k] = malloc(hight * sizeof(float**));
        for(j=0;j<hight;j++){
            CNN_weight[k][j] = malloc(howmany_filter * sizeof(float*));
            for(i=0;i<howmany_filter;i++){
               CNN_weight[k][j][i] = malloc(input_channel * sizeof(float));
            }
        }
    }
//m=0;m<input_channel;m++
//i=0;i<howmany_filter;i++

    for(i=0;i<howmany_filter;i++){
        for(m=0;m<input_channel;m++){
                                for(l=0;l<5;l++){
                                    fscanf(fn,"%c\n",&test22);
                                //    printf("%c",test22);
                                }
                                fscanf(fn,"%f\n",&read_in);
                               // printf("%f",read_in);

                                fscanf(fn,"%c\n",&test22);
                               // printf("%c",test22);

                                fscanf(fn,"%f\n",&read_in);
                               // printf("%f",read_in);
                                for(l=0;l<2;l++){
                                    fscanf(fn,"%c\n",&test22);
                                   // printf("%c",test22);
                                }


            test_count = test_count + 1;
           // printf("input channel %d   filter %d\n",input_channel,test_count);
            for(j=0;j<hight;j++){
                for(k=0;k<width;k++){
                    fscanf(fn,"%f\n",&read_in);
                    CNN_weight[k][j][i][m] = read_in;
                    //printf("f = %f\n",read_in);
                    //printf("%f",CNN_weight[k][j][i][m]);//3 3 50 1
                 //   printf("");
                }
               // printf("\n");
            }
           // printf("\n");
        }
    }
    //fclose(fn);
    return CNN_weight;

}

char**** load_CNN_weight_integer(int width, int hight, int input_channel, int howmany_filter,FILE* fn){
    char ****CNN_weight;
    int v;
    CNN_weight = malloc(width * sizeof(char***));
    for(int i = 0; i < width; i++){
		CNN_weight[i] = malloc(hight * sizeof(char**));
		for(int j = 0; j < hight; j++){
			CNN_weight[i][j] = malloc(input_channel * sizeof(char*));
			for(int k = 0; k < howmany_filter; k++)
				CNN_weight[i][j][k] = malloc(howmany_filter * sizeof(char));
		}
	}
	for(int n = 0; n < input_channel; n++){
		for(int i = 0; i < howmany_filter; i++){
			for(int j = 0; j < hight; j++){
				for(int k = 0; k < width; k++){
                    fscanf(fn, "%d ", &v);
					CNN_weight[k][j][n][i] = v;
				}
			}
        fscanf(fn, "\n");
		}
	}
/*
    for(int n = 0; n < howmany_filter; n++){
		for(int i = 0; i < input_channel; i++){
			for(int j = 0; j < hight; j++){
				for(int k = 0; k < width; k++){
					printf("%d",CNN_weight[k][j][i][n]);
				}
			}
        printf("\n");
		}
	}
*/
    return CNN_weight;
}

float** load_FC_weight_float(int input_channel_size, int classes, FILE* fn){
    float **FC_weight;
    int i,j;
    float read_in;
    int test_count=0;
    FC_weight = malloc(input_channel_size * sizeof(float*));
    for(i=0;i<input_channel_size;i++){
        FC_weight[i] = malloc(classes * sizeof(float));
    }
    for(i=0;i<input_channel_size;i++){
        for(j=0;j<classes;j++){
           test_count = test_count +1;
           fscanf(fn,"%f\n",&read_in);
           FC_weight[i][j] = read_in;//1800 270
          // printf("%hd\n",FC_weight[i][j]);
          // printf("test_count = %d\n",test_count);
        }

    }
    //printf("test_count = %d\n",test_count);

    return FC_weight;
}

float* load_FC_bias_float(int classes,FILE* fn){
	float *FC_bias;
	float read_in;
	FC_bias = malloc(classes * sizeof(float));

	for(int i = 0; i < classes; i++){
		fscanf(fn,"%f\n",&read_in);
		FC_bias[i] = read_in;
	}
	return FC_bias;
}

float* READ_bn_data(int size,FILE* fn){
    float* arrayy;
    float read_in;
    int i;
    arrayy = malloc(size * sizeof(float));
    for(i=0;i<size;i++){
        fscanf(fn,"%f\n",&read_in);
        arrayy[i] = read_in;
        //printf("%f\n",arrayy[i]);
    }

    return arrayy;
}

static void load_all_wt(void){
	FILE *fp;

	fp = fopen("../ws_resnet/conv1_W.txt","r");
    Conv1_float = load_CNN_weight_float(3,3,3,16,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv2blk1_bn_beta.txt","r");
    conv2blk1_bn_beta = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk1_bn_gamma.txt","r");
    conv2blk1_bn_gamma = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk1_bn_mean_EMA.txt","r");
    conv2blk1_bn_mean = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk1_bn_variance_EMA.txt","r");
    conv2blk1_bn_variance = READ_bn_data(16,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv2blk1_stembn_beta.txt","r");
    conv2blk1_stembn_beta = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk1_stembn_gamma.txt","r");
    conv2blk1_stembn_gamma = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk1_stembn_mean_EMA.txt","r");
    conv2blk1_stembn_mean = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk1_stembn_variance_EMA.txt","r");
    conv2blk1_stembn_variance = READ_bn_data(16,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv2blk1_c3x3a_W.txt","r");
    Conv2blk1_c3x3a_float = load_CNN_weight_float(3,3,16,16,fp);
	fp = fopen("../ws_resnet/conv2blk1_c3x3b_W.txt","r");
    Conv2blk1_c3x3b_float = load_CNN_weight_float(3,3,16,16,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv2blk2_bn_beta.txt","r");
    conv2blk2_bn_beta = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk2_bn_gamma.txt","r");
    conv2blk2_bn_gamma = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk2_bn_mean_EMA.txt","r");
    conv2blk2_bn_mean = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk2_bn_variance_EMA.txt","r");
    conv2blk2_bn_variance = READ_bn_data(16,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv2blk2_stembn_beta.txt","r");
    conv2blk2_stembn_beta = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk2_stembn_gamma.txt","r");
    conv2blk2_stembn_gamma = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk2_stembn_mean_EMA.txt","r");
    conv2blk2_stembn_mean = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk2_stembn_variance_EMA.txt","r");
    conv2blk2_stembn_variance = READ_bn_data(16,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv2blk2_c3x3a_W.txt","r");
    Conv2blk2_c3x3a_float = load_CNN_weight_float(3,3,16,16,fp);
	fp = fopen("../ws_resnet/conv2blk2_c3x3b_W.txt","r");
    Conv2blk2_c3x3b_float = load_CNN_weight_float(3,3,16,16,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv2blk3_bn_beta.txt","r");
    conv2blk3_bn_beta = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk3_bn_gamma.txt","r");
    conv2blk3_bn_gamma = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk3_bn_mean_EMA.txt","r");
    conv2blk3_bn_mean = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk3_bn_variance_EMA.txt","r");
    conv2blk3_bn_variance = READ_bn_data(16,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv2blk3_stembn_beta.txt","r");
    conv2blk3_stembn_beta = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk3_stembn_gamma.txt","r");
    conv2blk3_stembn_gamma = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk3_stembn_mean_EMA.txt","r");
    conv2blk3_stembn_mean = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv2blk3_stembn_variance_EMA.txt","r");
    conv2blk3_stembn_variance = READ_bn_data(16,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv2blk3_c3x3a_W.txt","r");
    Conv2blk3_c3x3a_float = load_CNN_weight_float(3,3,16,16,fp);
	fp = fopen("../ws_resnet/conv2blk3_c3x3b_W.txt","r");
    Conv2blk3_c3x3b_float = load_CNN_weight_float(3,3,16,16,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv3blk1_bn_beta.txt","r");
    conv3blk1_bn_beta = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv3blk1_bn_gamma.txt","r");
    conv3blk1_bn_gamma = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv3blk1_bn_mean_EMA.txt","r");
    conv3blk1_bn_mean = READ_bn_data(16,fp);
    fp = fopen("../ws_resnet/conv3blk1_bn_variance_EMA.txt","r");
    conv3blk1_bn_variance = READ_bn_data(16,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv3blk1_stembn_beta.txt","r");
    conv3blk1_stembn_beta = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk1_stembn_gamma.txt","r");
    conv3blk1_stembn_gamma = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk1_stembn_mean_EMA.txt","r");
    conv3blk1_stembn_mean = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk1_stembn_variance_EMA.txt","r");
    conv3blk1_stembn_variance = READ_bn_data(32,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv3blk1_c3x3a_W.txt","r");
    Conv3blk1_c3x3a_float = load_CNN_weight_float(3,3,16,32,fp);
	fp = fopen("../ws_resnet/conv3blk1_c3x3b_W.txt","r");
    Conv3blk1_c3x3b_float = load_CNN_weight_float(3,3,32,32,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv3blk2_bn_beta.txt","r");
    conv3blk2_bn_beta = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk2_bn_gamma.txt","r");
    conv3blk2_bn_gamma = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk2_bn_mean_EMA.txt","r");
    conv3blk2_bn_mean = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk2_bn_variance_EMA.txt","r");
    conv3blk2_bn_variance = READ_bn_data(32,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv3blk2_stembn_beta.txt","r");
    conv3blk2_stembn_beta = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk2_stembn_gamma.txt","r");
    conv3blk2_stembn_gamma = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk2_stembn_mean_EMA.txt","r");
    conv3blk2_stembn_mean = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk2_stembn_variance_EMA.txt","r");
    conv3blk2_stembn_variance = READ_bn_data(32,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv3blk2_c3x3a_W.txt","r");
    Conv3blk2_c3x3a_float = load_CNN_weight_float(3,3,32,32,fp);
	fp = fopen("../ws_resnet/conv3blk2_c3x3b_W.txt","r");
    Conv3blk2_c3x3b_float = load_CNN_weight_float(3,3,32,32,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv3blk3_bn_beta.txt","r");
    conv3blk3_bn_beta = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk3_bn_gamma.txt","r");
    conv3blk3_bn_gamma = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk3_bn_mean_EMA.txt","r");
    conv3blk3_bn_mean = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk3_bn_variance_EMA.txt","r");
    conv3blk3_bn_variance = READ_bn_data(32,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv3blk3_stembn_beta.txt","r");
    conv3blk3_stembn_beta = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk3_stembn_gamma.txt","r");
    conv3blk3_stembn_gamma = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk3_stembn_mean_EMA.txt","r");
    conv3blk3_stembn_mean = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv3blk3_stembn_variance_EMA.txt","r");
    conv3blk3_stembn_variance = READ_bn_data(32,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv3blk3_c3x3a_W.txt","r");
    Conv3blk3_c3x3a_float = load_CNN_weight_float(3,3,32,32,fp);
	fp = fopen("../ws_resnet/conv3blk3_c3x3b_W.txt","r");
    Conv3blk3_c3x3b_float = load_CNN_weight_float(3,3,32,32,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv4blk1_bn_beta.txt","r");
    conv4blk1_bn_beta = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv4blk1_bn_gamma.txt","r");
    conv4blk1_bn_gamma = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv4blk1_bn_mean_EMA.txt","r");
    conv4blk1_bn_mean = READ_bn_data(32,fp);
    fp = fopen("../ws_resnet/conv4blk1_bn_variance_EMA.txt","r");
    conv4blk1_bn_variance = READ_bn_data(32,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv4blk1_stembn_beta.txt","r");
    conv4blk1_stembn_beta = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk1_stembn_gamma.txt","r");
    conv4blk1_stembn_gamma = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk1_stembn_mean_EMA.txt","r");
    conv4blk1_stembn_mean = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk1_stembn_variance_EMA.txt","r");
    conv4blk1_stembn_variance = READ_bn_data(64,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv4blk1_c3x3a_W.txt","r");
    Conv4blk1_c3x3a_float = load_CNN_weight_float(3,3,32,64,fp);
	fp = fopen("../ws_resnet/conv4blk1_c3x3b_W.txt","r");
    Conv4blk1_c3x3b_float = load_CNN_weight_float(3,3,64,64,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv4blk2_bn_beta.txt","r");
    conv4blk2_bn_beta = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk2_bn_gamma.txt","r");
    conv4blk2_bn_gamma = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk2_bn_mean_EMA.txt","r");
    conv4blk2_bn_mean = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk2_bn_variance_EMA.txt","r");
    conv4blk2_bn_variance = READ_bn_data(64,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv4blk2_stembn_beta.txt","r");
    conv4blk2_stembn_beta = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk2_stembn_gamma.txt","r");
    conv4blk2_stembn_gamma = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk2_stembn_mean_EMA.txt","r");
    conv4blk2_stembn_mean = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk2_stembn_variance_EMA.txt","r");
    conv4blk2_stembn_variance = READ_bn_data(64,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv4blk2_c3x3a_W.txt","r");
    Conv4blk2_c3x3a_float = load_CNN_weight_float(3,3,64,64,fp);
	fp = fopen("../ws_resnet/conv4blk2_c3x3b_W.txt","r");
    Conv4blk2_c3x3b_float = load_CNN_weight_float(3,3,64,64,fp);

	////////////load bn data///////////////
    fp = fopen("../ws_resnet/conv4blk3_bn_beta.txt","r");
    conv4blk3_bn_beta = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk3_bn_gamma.txt","r");
    conv4blk3_bn_gamma = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk3_bn_mean_EMA.txt","r");
    conv4blk3_bn_mean = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk3_bn_variance_EMA.txt","r");
    conv4blk3_bn_variance = READ_bn_data(64,fp);
	////////////load stembn data///////////////
	fp = fopen("../ws_resnet/conv4blk3_stembn_beta.txt","r");
    conv4blk3_stembn_beta = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk3_stembn_gamma.txt","r");
    conv4blk3_stembn_gamma = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk3_stembn_mean_EMA.txt","r");
    conv4blk3_stembn_mean = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/conv4blk3_stembn_variance_EMA.txt","r");
    conv4blk3_stembn_variance = READ_bn_data(64,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/conv4blk3_c3x3a_W.txt","r");
    Conv4blk3_c3x3a_float = load_CNN_weight_float(3,3,64,64,fp);
	fp = fopen("../ws_resnet/conv4blk3_c3x3b_W.txt","r");
    Conv4blk3_c3x3b_float = load_CNN_weight_float(3,3,64,64,fp);

	////////////load lastbn data///////////////
	fp = fopen("../ws_resnet/lastbn_beta.txt","r");
    lastbn_beta = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/lastbn_gamma.txt","r");
    lastbn_gamma = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/lastbn_mean_EMA.txt","r");
    lastbn_mean = READ_bn_data(64,fp);
    fp = fopen("../ws_resnet/lastbn_variance_EMA.txt","r");
    lastbn_variance = READ_bn_data(64,fp);
	//////////load weight data/////////////
	fp = fopen("../ws_resnet/fct_b.txt","r");
    fct_b = load_FC_bias_float(10,fp);
	fp = fopen("../ws_resnet/fct_W.txt","r");
    fct_W = load_FC_weight_float(64,10,fp);

	fclose(fp);
}

static void load_all_wt_resnet_float(void){
	FILE *fp;

	fp = fopen("conv1_W.txt","r");
    Conv1_float = load_CNN_weight_float(3,3,3,16,fp);

	////////////load bn data///////////////
    fp = fopen("conv2blk1_bn_beta.txt","r");
    conv2blk1_bn_beta = READ_bn_data(16,fp);
    fp = fopen("conv2blk1_bn_gamma.txt","r");
    conv2blk1_bn_gamma = READ_bn_data(16,fp);
    fp = fopen("conv2blk1_bn_mean_EMA.txt","r");
    conv2blk1_bn_mean = READ_bn_data(16,fp);
    fp = fopen("conv2blk1_bn_variance_EMA.txt","r");
    conv2blk1_bn_variance = READ_bn_data(16,fp);
	////////////load stembn data///////////////
	fp = fopen("conv2blk1_stembn_beta.txt","r");
    conv2blk1_stembn_beta = READ_bn_data(16,fp);
    fp = fopen("conv2blk1_stembn_gamma.txt","r");
    conv2blk1_stembn_gamma = READ_bn_data(16,fp);
    fp = fopen("conv2blk1_stembn_mean_EMA.txt","r");
    conv2blk1_stembn_mean = READ_bn_data(16,fp);
    fp = fopen("conv2blk1_stembn_variance_EMA.txt","r");
    conv2blk1_stembn_variance = READ_bn_data(16,fp);
	//////////load weight data/////////////
	fp = fopen("conv2blk1_c3x3a_W.txt","r");
    Conv2blk1_c3x3a_float = load_CNN_weight_float(3,3,16,16,fp);
	fp = fopen("conv2blk1_c3x3b_W.txt","r");
    Conv2blk1_c3x3b_float = load_CNN_weight_float(3,3,16,16,fp);

	////////////load bn data///////////////
    fp = fopen("conv2blk2_bn_beta.txt","r");
    conv2blk2_bn_beta = READ_bn_data(16,fp);
    fp = fopen("conv2blk2_bn_gamma.txt","r");
    conv2blk2_bn_gamma = READ_bn_data(16,fp);
    fp = fopen("conv2blk2_bn_mean_EMA.txt","r");
    conv2blk2_bn_mean = READ_bn_data(16,fp);
    fp = fopen("conv2blk2_bn_variance_EMA.txt","r");
    conv2blk2_bn_variance = READ_bn_data(16,fp);
	////////////load stembn data///////////////
	fp = fopen("conv2blk2_stembn_beta.txt","r");
    conv2blk2_stembn_beta = READ_bn_data(16,fp);
    fp = fopen("conv2blk2_stembn_gamma.txt","r");
    conv2blk2_stembn_gamma = READ_bn_data(16,fp);
    fp = fopen("conv2blk2_stembn_mean_EMA.txt","r");
    conv2blk2_stembn_mean = READ_bn_data(16,fp);
    fp = fopen("conv2blk2_stembn_variance_EMA.txt","r");
    conv2blk2_stembn_variance = READ_bn_data(16,fp);
	//////////load weight data/////////////
	fp = fopen("conv2blk2_c3x3a_W.txt","r");
    Conv2blk2_c3x3a_float = load_CNN_weight_float(3,3,16,16,fp);
	fp = fopen("conv2blk2_c3x3b_W.txt","r");
    Conv2blk2_c3x3b_float = load_CNN_weight_float(3,3,16,16,fp);

	////////////load bn data///////////////
    fp = fopen("conv2blk3_bn_beta.txt","r");
    conv2blk3_bn_beta = READ_bn_data(16,fp);
    fp = fopen("conv2blk3_bn_gamma.txt","r");
    conv2blk3_bn_gamma = READ_bn_data(16,fp);
    fp = fopen("conv2blk3_bn_mean_EMA.txt","r");
    conv2blk3_bn_mean = READ_bn_data(16,fp);
    fp = fopen("conv2blk3_bn_variance_EMA.txt","r");
    conv2blk3_bn_variance = READ_bn_data(16,fp);
	////////////load stembn data///////////////
	fp = fopen("conv2blk3_stembn_beta.txt","r");
    conv2blk3_stembn_beta = READ_bn_data(16,fp);
    fp = fopen("conv2blk3_stembn_gamma.txt","r");
    conv2blk3_stembn_gamma = READ_bn_data(16,fp);
    fp = fopen("conv2blk3_stembn_mean_EMA.txt","r");
    conv2blk3_stembn_mean = READ_bn_data(16,fp);
    fp = fopen("conv2blk3_stembn_variance_EMA.txt","r");
    conv2blk3_stembn_variance = READ_bn_data(16,fp);
	//////////load weight data/////////////
	fp = fopen("conv2blk3_c3x3a_W.txt","r");
    Conv2blk3_c3x3a_float = load_CNN_weight_float(3,3,16,16,fp);
	fp = fopen("conv2blk3_c3x3b_W.txt","r");
    Conv2blk3_c3x3b_float = load_CNN_weight_float(3,3,16,16,fp);

	////////////load bn data///////////////
    fp = fopen("conv3blk1_bn_beta.txt","r");
    conv3blk1_bn_beta = READ_bn_data(16,fp);
    fp = fopen("conv3blk1_bn_gamma.txt","r");
    conv3blk1_bn_gamma = READ_bn_data(16,fp);
    fp = fopen("conv3blk1_bn_mean_EMA.txt","r");
    conv3blk1_bn_mean = READ_bn_data(16,fp);
    fp = fopen("conv3blk1_bn_variance_EMA.txt","r");
    conv3blk1_bn_variance = READ_bn_data(16,fp);
	////////////load stembn data///////////////
	fp = fopen("conv3blk1_stembn_beta.txt","r");
    conv3blk1_stembn_beta = READ_bn_data(32,fp);
    fp = fopen("conv3blk1_stembn_gamma.txt","r");
    conv3blk1_stembn_gamma = READ_bn_data(32,fp);
    fp = fopen("conv3blk1_stembn_mean_EMA.txt","r");
    conv3blk1_stembn_mean = READ_bn_data(32,fp);
    fp = fopen("conv3blk1_stembn_variance_EMA.txt","r");
    conv3blk1_stembn_variance = READ_bn_data(32,fp);
	//////////load weight data/////////////
	fp = fopen("conv3blk1_c3x3a_W.txt","r");
    Conv3blk1_c3x3a_float = load_CNN_weight_float(3,3,16,32,fp);
	fp = fopen("conv3blk1_c3x3b_W.txt","r");
    Conv3blk1_c3x3b_float = load_CNN_weight_float(3,3,32,32,fp);

	////////////load bn data///////////////
    fp = fopen("conv3blk2_bn_beta.txt","r");
    conv3blk2_bn_beta = READ_bn_data(32,fp);
    fp = fopen("conv3blk2_bn_gamma.txt","r");
    conv3blk2_bn_gamma = READ_bn_data(32,fp);
    fp = fopen("conv3blk2_bn_mean_EMA.txt","r");
    conv3blk2_bn_mean = READ_bn_data(32,fp);
    fp = fopen("conv3blk2_bn_variance_EMA.txt","r");
    conv3blk2_bn_variance = READ_bn_data(32,fp);
	////////////load stembn data///////////////
	fp = fopen("conv3blk2_stembn_beta.txt","r");
    conv3blk2_stembn_beta = READ_bn_data(32,fp);
    fp = fopen("conv3blk2_stembn_gamma.txt","r");
    conv3blk2_stembn_gamma = READ_bn_data(32,fp);
    fp = fopen("conv3blk2_stembn_mean_EMA.txt","r");
    conv3blk2_stembn_mean = READ_bn_data(32,fp);
    fp = fopen("conv3blk2_stembn_variance_EMA.txt","r");
    conv3blk2_stembn_variance = READ_bn_data(32,fp);
	//////////load weight data/////////////
	fp = fopen("conv3blk2_c3x3a_W.txt","r");
    Conv3blk2_c3x3a_float = load_CNN_weight_float(3,3,32,32,fp);
	fp = fopen("conv3blk2_c3x3b_W.txt","r");
    Conv3blk2_c3x3b_float = load_CNN_weight_float(3,3,32,32,fp);

	////////////load bn data///////////////
    fp = fopen("conv3blk3_bn_beta.txt","r");
    conv3blk3_bn_beta = READ_bn_data(32,fp);
    fp = fopen("conv3blk3_bn_gamma.txt","r");
    conv3blk3_bn_gamma = READ_bn_data(32,fp);
    fp = fopen("conv3blk3_bn_mean_EMA.txt","r");
    conv3blk3_bn_mean = READ_bn_data(32,fp);
    fp = fopen("conv3blk3_bn_variance_EMA.txt","r");
    conv3blk3_bn_variance = READ_bn_data(32,fp);
	////////////load stembn data///////////////
	fp = fopen("conv3blk3_stembn_beta.txt","r");
    conv3blk3_stembn_beta = READ_bn_data(32,fp);
    fp = fopen("conv3blk3_stembn_gamma.txt","r");
    conv3blk3_stembn_gamma = READ_bn_data(32,fp);
    fp = fopen("conv3blk3_stembn_mean_EMA.txt","r");
    conv3blk3_stembn_mean = READ_bn_data(32,fp);
    fp = fopen("conv3blk3_stembn_variance_EMA.txt","r");
    conv3blk3_stembn_variance = READ_bn_data(32,fp);
	//////////load weight data/////////////
	fp = fopen("conv3blk3_c3x3a_W.txt","r");
    Conv3blk3_c3x3a_float = load_CNN_weight_float(3,3,32,32,fp);
	fp = fopen("conv3blk3_c3x3b_W.txt","r");
    Conv3blk3_c3x3b_float = load_CNN_weight_float(3,3,32,32,fp);

	////////////load bn data///////////////
    fp = fopen("conv4blk1_bn_beta.txt","r");
    conv4blk1_bn_beta = READ_bn_data(32,fp);
    fp = fopen("conv4blk1_bn_gamma.txt","r");
    conv4blk1_bn_gamma = READ_bn_data(32,fp);
    fp = fopen("conv4blk1_bn_mean_EMA.txt","r");
    conv4blk1_bn_mean = READ_bn_data(32,fp);
    fp = fopen("conv4blk1_bn_variance_EMA.txt","r");
    conv4blk1_bn_variance = READ_bn_data(32,fp);
	////////////load stembn data///////////////
	fp = fopen("conv4blk1_stembn_beta.txt","r");
    conv4blk1_stembn_beta = READ_bn_data(64,fp);
    fp = fopen("conv4blk1_stembn_gamma.txt","r");
    conv4blk1_stembn_gamma = READ_bn_data(64,fp);
    fp = fopen("conv4blk1_stembn_mean_EMA.txt","r");
    conv4blk1_stembn_mean = READ_bn_data(64,fp);
    fp = fopen("conv4blk1_stembn_variance_EMA.txt","r");
    conv4blk1_stembn_variance = READ_bn_data(64,fp);
	//////////load weight data/////////////
	fp = fopen("conv4blk1_c3x3a_W.txt","r");
    Conv4blk1_c3x3a_float = load_CNN_weight_float(3,3,32,64,fp);
	fp = fopen("conv4blk1_c3x3b_W.txt","r");
    Conv4blk1_c3x3b_float = load_CNN_weight_float(3,3,64,64,fp);

	////////////load bn data///////////////
    fp = fopen("conv4blk2_bn_beta.txt","r");
    conv4blk2_bn_beta = READ_bn_data(64,fp);
    fp = fopen("conv4blk2_bn_gamma.txt","r");
    conv4blk2_bn_gamma = READ_bn_data(64,fp);
    fp = fopen("conv4blk2_bn_mean_EMA.txt","r");
    conv4blk2_bn_mean = READ_bn_data(64,fp);
    fp = fopen("conv4blk2_bn_variance_EMA.txt","r");
    conv4blk2_bn_variance = READ_bn_data(64,fp);
	////////////load stembn data///////////////
	fp = fopen("conv4blk2_stembn_beta.txt","r");
    conv4blk2_stembn_beta = READ_bn_data(64,fp);
    fp = fopen("conv4blk2_stembn_gamma.txt","r");
    conv4blk2_stembn_gamma = READ_bn_data(64,fp);
    fp = fopen("conv4blk2_stembn_mean_EMA.txt","r");
    conv4blk2_stembn_mean = READ_bn_data(64,fp);
    fp = fopen("conv4blk2_stembn_variance_EMA.txt","r");
    conv4blk2_stembn_variance = READ_bn_data(64,fp);
	//////////load weight data/////////////
	fp = fopen("conv4blk2_c3x3a_W.txt","r");
    Conv4blk2_c3x3a_float = load_CNN_weight_float(3,3,64,64,fp);
	fp = fopen("conv4blk2_c3x3b_W.txt","r");
    Conv4blk2_c3x3b_float = load_CNN_weight_float(3,3,64,64,fp);

	////////////load bn data///////////////
    fp = fopen("conv4blk3_bn_beta.txt","r");
    conv4blk3_bn_beta = READ_bn_data(64,fp);
    fp = fopen("conv4blk3_bn_gamma.txt","r");
    conv4blk3_bn_gamma = READ_bn_data(64,fp);
    fp = fopen("conv4blk3_bn_mean_EMA.txt","r");
    conv4blk3_bn_mean = READ_bn_data(64,fp);
    fp = fopen("conv4blk3_bn_variance_EMA.txt","r");
    conv4blk3_bn_variance = READ_bn_data(64,fp);
	////////////load stembn data///////////////
	fp = fopen("conv4blk3_stembn_beta.txt","r");
    conv4blk3_stembn_beta = READ_bn_data(64,fp);
    fp = fopen("conv4blk3_stembn_gamma.txt","r");
    conv4blk3_stembn_gamma = READ_bn_data(64,fp);
    fp = fopen("conv4blk3_stembn_mean_EMA.txt","r");
    conv4blk3_stembn_mean = READ_bn_data(64,fp);
    fp = fopen("conv4blk3_stembn_variance_EMA.txt","r");
    conv4blk3_stembn_variance = READ_bn_data(64,fp);
	//////////load weight data/////////////
	fp = fopen("conv4blk3_c3x3a_W.txt","r");
    Conv4blk3_c3x3a_float = load_CNN_weight_float(3,3,64,64,fp);
	fp = fopen("conv4blk3_c3x3b_W.txt","r");
    Conv4blk3_c3x3b_float = load_CNN_weight_float(3,3,64,64,fp);

	////////////load lastbn data///////////////
	fp = fopen("lastbn_beta.txt","r");
    lastbn_beta = READ_bn_data(64,fp);
    fp = fopen("lastbn_gamma.txt","r");
    lastbn_gamma = READ_bn_data(64,fp);
    fp = fopen("lastbn_mean_EMA.txt","r");
    lastbn_mean = READ_bn_data(64,fp);
    fp = fopen("lastbn_variance_EMA.txt","r");
    lastbn_variance = READ_bn_data(64,fp);
	//////////load weight data/////////////
	fp = fopen("fct_b.txt","r");
    fct_b = load_FC_bias_float(10,fp);
	fp = fopen("fct_W.txt","r");
    fct_W = load_FC_weight_float(64,10,fp);

	fclose(fp);
}

float*** INPUT_minus_mean(float*** input,float*** one,float*** two,float*** three,int size){
    int i,j,k;
	float*** OUT_array;
    OUT_array = malloc(size * sizeof(float**));
    for(i=0;i<size;i++){
        OUT_array[i] = malloc(size * sizeof(float*));
        for(j=0;j<size;j++){
            OUT_array[i][j] = malloc(3 * sizeof(float));
        }
    }
    for(i=0;i<3;i++){
        for(j=0;j<size;j++){
            for(k=0;k<size;k++){
                    if(i==0)
                        OUT_array[k][j][i] =  (input[k][j][i] - one[k][j][0])/128;
                    else if(i==1)
                        OUT_array[k][j][i] =  (input[k][j][i] - two[k][j][0])/128;
                    else
                        OUT_array[k][j][i] =  (input[k][j][i] - three[k][j][0])/128;
            }
        }
    }
    return OUT_array;
}

float**** CON_function_float(float*** input,float**** in_filter,int in_channel,int input_size,int filter_col,int filter_size,int stride){
    int i,j,k,l,w,h;
    int how_many_stride;

    int rrrr;
    float**** output_arry;
    float temp;
    how_many_stride = (input_size - filter_size)/stride + 1;


    output_arry = malloc(how_many_stride * sizeof(float***));
    for(k=0;k<how_many_stride;k++){
        output_arry[k] = malloc(how_many_stride * sizeof(float**));
        for(j=0;j<how_many_stride;j++){
            output_arry[k][j] = malloc(in_channel * sizeof(float*));
            for(i=0;i<in_channel;i++){
               output_arry[k][j][i] = malloc(filter_col * sizeof(float));
            }
        }
    }
    //initial
    for(l=0;l<filter_col;l++){
        for(k=0;k<in_channel;k++){
            for(j=0;j<how_many_stride;j++){
                for(i=0;i<how_many_stride;i++){
                    output_arry[i][j][k][l] = 0;
                }
            }
        }
    }

    //input[input_size][input_size][in_channel];
    //in_filter[filter_size][filter_size][in_channel][filter_col];
    //input_size - 3 + 1
    for(l=0;l<filter_col;l++){
        for(k=0;k<in_channel;k++){
            for(h=0;h<how_many_stride;h++){
                for(w=0;w<how_many_stride;w++){
                    for(j=0;j<filter_size;j++){
                        for(i=0;i<filter_size;i++){
                            temp = in_filter[i][j][k][l] * input[i+(w*stride)][j+(h*stride)][k];
                            output_arry[w][h][k][l] = output_arry[w][h][k][l] + temp;
                        }
                    }
                }
            }
        }
    }

    return output_arry;
}

float*** CNN_combine_float(float**** CNN_out_array,int sizee, int channel, int col){
    int i,j,k,l;
    float*** OUT_array;

    OUT_array = malloc(sizee * sizeof(float**));
    for(i=0;i<sizee;i++){
        OUT_array[i] = malloc(sizee * sizeof(float*));
        for(j=0;j<sizee;j++){
            OUT_array[i][j] = malloc(col * sizeof(float));
        }
    }
    for(i=0;i<col;i++){
        for(j=0;j<sizee;j++){
            for(k=0;k<sizee;k++){
                OUT_array[k][j][i] = 0;
            }
        }
    }
    for(i=0;i<col;i++){
        for(j=0;j<channel;j++){
            for(k=0;k<sizee;k++){
                for(l=0;l<sizee;l++){
                     OUT_array[l][k][i] = CNN_out_array[l][k][j][i] + OUT_array[l][k][i];
                }
            }
        }
    }
    return OUT_array;
}

float* FC_FUNCTION_float(float* inputt,float** FC_weight,float* bias,int left_size, int right_size){
    int i,j;
    float* FU_out;
    FU_out = malloc(right_size * sizeof(float));
    for(i=0;i<right_size;i++){
        FU_out[i] = 0;
    }

    for(i=0;i<right_size;i++){
        for(j=0;j<left_size;j++){
            FU_out[i] = FU_out[i] + FC_weight[j][i]*inputt[j];
        }
        FU_out[i] = FU_out[i] + bias[i];
    }
    return FU_out;
}

float*** PAD(float*** input,int size,int channel){
    float*** input_expend;
    int i,j,k;
    input_expend = malloc((size+2) * sizeof(float**));
    for(i=0;i<(size+2);i++){
          input_expend[i] =  malloc((size+2) * sizeof(float*));
        for(j=0;j<(size+2);j++){
            input_expend[i][j] =  malloc(channel * sizeof(float));
        }
    }

    for(k=0;k<channel;k++){

        for(i=0;i<(size+2);i++)
            input_expend[i][size+1][k]=0;
        for(i=0;i<(size+2);i++)
            input_expend[i][0][k]=0;
        for(i=0;i<(size+2);i++)
            input_expend[size+1][i][k]=0;
        for(i=0;i<(size+2);i++)
            input_expend[0][i][k]=0;

        for(i=1;i<(size+1);i++){
            for(j=1;j<(size+1);j++)
                input_expend[j][i][k]=input[j-1][i-1][k];
        }
    }
    return input_expend;
}

float*** PAD_1(float*** input,int size,int channel){
    float*** input_expend;
    int i,j,k;
    input_expend = malloc((size+1) * sizeof(float**));
    for(i=0;i<(size+1);i++){
          input_expend[i] =  malloc((size+1) * sizeof(float*));
        for(j=0;j<(size+1);j++){
            input_expend[i][j] =  malloc(channel * sizeof(float));
        }
    }

    for(k=0;k<channel;k++){

        for(i=0;i<(size+1);i++)
            input_expend[i][size][k]=0;
        for(i=0;i<(size+1);i++)
            input_expend[size][i][k]=0;

        for(i=0;i<size;i++){
            for(j=0;j<size;j++){
                input_expend[j][i][k]=input[j][i][k];
            }
        }
    }
    return input_expend;
}

float*** BN_function(float*** conv_input,float* beta,float* gamma,float* variance,float* mean,int size,int channel){
    float*** BN_out_array;
    int i,j,k;

    BN_out_array = malloc(size * sizeof(float**));
    for(i=0;i<size;i++){
        BN_out_array[i] = malloc(size * sizeof(float*));
        for(j=0;j<size;j++){
            BN_out_array[i][j] = malloc(channel * sizeof(float));
        }
    }

    for(i=0;i<channel;i++){
        for(j=0;j<size;j++){
            for(k=0;k<size;k++){
                BN_out_array[k][j][i] = (((conv_input[k][j][i] - mean[i])/sqrt(variance[i]))*gamma[i]) + beta[i];
            }
        }
    }
    return BN_out_array;

}

float*** ACT_float(float*** bn_input,int size,int channel){
    int i,j,k;
    for(i=0;i<channel;i++){
        for(j=0;j<size;j++){
            for(k=0;k<size;k++){

                if(bn_input[k][j][i] < 0){
                    bn_input[k][j][i] = 0;
                }
                else if( (bn_input[k][j][i]) >= 1){
                    bn_input[k][j][i] = 3;
                }
                    else{
                        bn_input[k][j][i] = bn_input[k][j][i] * 3;

                if( (bn_input[k][j][i]) >= 2.5){
                    bn_input[k][j][i] = 3;
                }
                else if( ((bn_input[k][j][i]) < 2.5) && ((bn_input[k][j][i]) >= 1.5)   ){
                    bn_input[k][j][i] = 2;
                }
                 else if( ((bn_input[k][j][i]) < 1.5) && ((bn_input[k][j][i]) >= 0.5)   ){
                    bn_input[k][j][i] = 1;
                }
                else{
                    bn_input[k][j][i] = 0;
                }
                    }
                bn_input[k][j][i] = bn_input[k][j][i]/3;
            }
        }
    }
    return bn_input;
}

float*** ACT_float_relu(float*** bn_input,int size,int channel){
    int i,j,k;
    for(i=0;i<channel;i++){
        for(j=0;j<size;j++){
            for(k=0;k<size;k++){
                if(bn_input[k][j][i] < 0)
                    bn_input[k][j][i] = 0;
                else
                    bn_input[k][j][i] = bn_input[k][j][i];
            }
        }
    }
    return bn_input;
}

float ***Average_pooling(float ***inArray, int size, int inChannel){
	float ***outArray;
	outArray = calloc(size/2,sizeof(float**));
	for(int i = 0; i < size/2; i++){
		outArray[i] = calloc(size/2,sizeof(float*));
		for(int j = 0; j < size/2; j++)
			outArray[i][j] = calloc(inChannel,sizeof(float));
	}
	for (int cha_in = 0; cha_in < inChannel; cha_in++){
		for (int row = 0; row < (size/2); row++){
			for (int col = 0; col < (size/2); col++){
				outArray[col][row][cha_in] = (inArray[col*2][row*2][cha_in]+inArray[col*2][row*2+1][cha_in]
				+inArray[col*2+1][row*2][cha_in]+inArray[col*2+1][row*2+1][cha_in])/4;
			}
		}
	}

	return outArray;
}

float*** Channel_PADDING(float ***inArray, int size, int inChannel){
	float ***outArray;
	outArray = calloc(size,sizeof(float**));
	for(int i = 0; i < size; i++){
		outArray[i] = calloc(size,sizeof(float*));
		for(int j = 0; j < size; j++)
			outArray[i][j] = calloc(inChannel*2,sizeof(float));
	}

	for (int cha_in = 0; cha_in < inChannel*2; cha_in++){
		for (int row = 0; row < size; row++){
			for (int col = 0; col < size; col++){
				if((cha_in < inChannel/2) || (cha_in >= inChannel*3/2))
					outArray[col][row][cha_in] = 0;
				else
					outArray[col][row][cha_in] = inArray[col][row][cha_in-(inChannel/2)];
			}
		}
	}

	return outArray;
}

float*** TWO_array_plus(float*** input_one,float*** input_two,int size,int channel){
    int i,j,k;
    for(i=0;i<channel;i++){
        for(j=0;j<size;j++){
            for(k=0;k<size;k++){
                //////////////////////////////////////need free input_two !!!!!!!!!!!!!
                input_one[k][j][i] = input_one[k][j][i] + input_two[k][j][i];
                //////////////////////////////////////need free input_two !!!!!!!!!!!!!
            }
        }
    }
    return input_one;
}

float* I_AM_SO_HAPPY_BECAUSE_IT_IS_THE_END_AVERGE(float*** input_array,int size,int Channel){
    float* Average;
    int i,j,k;
    int sizzee;
    sizzee = size*size;
    Average = malloc(Channel * sizeof(float));
    for(i=0;i<Channel;i++){
        Average[i]=0;
    }

    for(i=0;i<Channel;i++){
        for(j=0;j<size;j++){
            for(k=0;k<size;k++){
                Average[i] = Average[i] + input_array[k][j][i];
            }
        }
        Average[i] = Average[i]/sizzee;
    }
    return Average;
}

void free_float_3d(float*** array,int size){
    for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
            free(array[i][j]);
		}
		free(array[i]);
	}
	free(array);
	//printf("%x\n",array);
}

void free_float_4d(float**** array,int size,int inChannel){
    for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
            for(int k = 0; k < inChannel; k++){
                free(array[i][j][k]);
            }
            free(array[i][j]);
		}
		free(array[i]);
	}
	free(array);
}

float ***Convblk(int inSize,int inChannel,int outChannel,float ***inArray,float* beta,float* gamma,float* variance,float* mean,float**** filter_c3x3a,
float* stembeta,float* stemgamma,float* stemvariance,float* stemmean,float**** filter_c3x3b){

	float ***Convblk_out;
	float ***Convblk_bn;
	float ***Convblk_bn_pad;
	float ***Convblk_stembn;
	float ***Convblk_stembn_pad;
	float ***interArray;
	float ****Convblk_c3x3a_output;
	float ****Convblk_c3x3b_output;
	//FILE *fr;
	//fr = fopen("outputblk.txt","w");

	Convblk_bn = BN_function(inArray,beta,gamma,variance,mean,inSize,inChannel);
	Convblk_bn = ACT_float(Convblk_bn,inSize,inChannel);
	Convblk_bn_pad = PAD(Convblk_bn,inSize,inChannel);
	Convblk_c3x3a_output = CON_function_float(Convblk_bn_pad,filter_c3x3a,inChannel,inSize+2,outChannel,3,1);
	interArray = CNN_combine_float(Convblk_c3x3a_output,inSize,inChannel,outChannel);

	//interArray = TWO_array_plus(interArray,inArray,inSize,inChannel);

	Convblk_stembn = BN_function(interArray,stembeta,stemgamma,stemvariance,stemmean,inSize,inChannel);
	Convblk_stembn = ACT_float(Convblk_stembn,inSize,inChannel);
	Convblk_stembn_pad = PAD(Convblk_stembn,inSize,inChannel);
	Convblk_c3x3b_output = CON_function_float(Convblk_stembn_pad,filter_c3x3b,inChannel,inSize+2,outChannel,3,1);
	Convblk_out = CNN_combine_float(Convblk_c3x3b_output,inSize,inChannel,outChannel);

	Convblk_out = TWO_array_plus(Convblk_out,inArray,inSize,inChannel);

    free_float_3d(Convblk_bn,inSize);
    free_float_3d(Convblk_bn_pad,inSize+2);
	free_float_4d(Convblk_c3x3a_output,inSize,inChannel);

	free_float_3d(interArray,inSize);

    free_float_3d(Convblk_stembn,inSize);
    free_float_3d(Convblk_stembn_pad,inSize+2);
	free_float_4d(Convblk_c3x3b_output,inSize,inChannel);

	return Convblk_out;
}

float ***Convblk_stride2(int inSize,int inChannel,int outChannel,float ***inArray,float* beta,float* gamma,float* variance,float* mean,float**** filter_c3x3a,
float* stembeta,float* stemgamma,float* stemvariance,float* stemmean,float**** filter_c3x3b){

	float ***Convblk_out;
	float ***Convblk_bn;
	float ***Convblk_bn_pad;
	float ***Convblk_stembn;
	float ***Convblk_stembn_pad;
	float ***interArray;
	float ****Convblk_c3x3a_output;
	float ****Convblk_c3x3b_output;

	float ***Average_array;
	float ***Padding_array;

    //FILE *fr;
    //fr = fopen("output_Convblk.txt","w");

	Convblk_bn = BN_function(inArray,beta,gamma,variance,mean,inSize,inChannel);
	Convblk_bn = ACT_float(Convblk_bn,inSize,inChannel);
	Convblk_bn_pad = PAD_1(Convblk_bn,inSize,inChannel);
	Convblk_c3x3a_output = CON_function_float(Convblk_bn_pad,filter_c3x3a,inChannel,inSize+1,outChannel,3,2);
	interArray = CNN_combine_float(Convblk_c3x3a_output,inSize/2,inChannel,outChannel);

	Average_array = Average_pooling(inArray,inSize,inChannel);
	Padding_array = Channel_PADDING(Average_array,inSize/2,inChannel);

	//interArray = TWO_array_plus(interArray,Padding_array,inSize/2,outChannel);

	Convblk_stembn = BN_function(interArray,stembeta,stemgamma,stemvariance,stemmean,inSize/2,outChannel);
	Convblk_stembn = ACT_float(Convblk_stembn,inSize/2,outChannel);
	Convblk_stembn_pad = PAD(Convblk_stembn,inSize/2,outChannel);
	Convblk_c3x3b_output = CON_function_float(Convblk_stembn_pad,filter_c3x3b,outChannel,inSize/2+2,outChannel,3,1);
	Convblk_out = CNN_combine_float(Convblk_c3x3b_output,inSize/2,outChannel,outChannel);

	Convblk_out = TWO_array_plus(Convblk_out,Padding_array,inSize/2,outChannel);

    free_float_3d(Convblk_bn,inSize);
    free_float_3d(Convblk_bn_pad,inSize+1);
	free_float_4d(Convblk_c3x3a_output,inSize/2,inChannel);

	free_float_3d(interArray,inSize/2);
	free_float_3d(Average_array,inSize/2);
	free_float_3d(Padding_array,inSize/2);

    free_float_3d(Convblk_stembn,inSize/2);
    free_float_3d(Convblk_stembn_pad,inSize/2+2);
	free_float_4d(Convblk_c3x3b_output,inSize/2,inChannel);

	return Convblk_out;
}

float *softmax (float *fc_output){
    for(int i = 0; i < 10; i++){
        fc_output[i] = exp(fc_output[i])/(exp(fc_output[0])+exp(fc_output[1])+exp(fc_output[2])
        +exp(fc_output[3])+exp(fc_output[4])+exp(fc_output[5])+exp(fc_output[6])+exp(fc_output[7])
        +exp(fc_output[8])+exp(fc_output[9]));
    }
    return fc_output;
}



int classify(float***input)
{
    FILE *fr;
    fr = fopen("output.txt","w");
	int outcome = 0;

	float ***input_minus_mean;
	float ***input_pad;
	float ****Conv1_output;
	float ***Conv1_combine;

	float ***Conv2blk1_c3x3b_combine;
	float ***Conv2blk2_c3x3b_combine;
	float ***Conv2blk3_c3x3b_combine;

    float ***Conv3blk1_c3x3b_combine;
	float ***Conv3blk2_c3x3b_combine;
	float ***Conv3blk3_c3x3b_combine;

	float ***Conv4blk1_c3x3b_combine;
	float ***Conv4blk2_c3x3b_combine;
	float ***Conv4blk3_c3x3b_combine;

    float ***lastbn;
    float *lastbn_average;
    float *fc_output;



	input_minus_mean = INPUT_minus_mean(input,input_mean_0,input_mean_1,input_mean_2,32);
	input_pad = PAD(input_minus_mean,32,3);  //needed free
	Conv1_output = CON_function_float(input_pad,Conv1_float,3,34,16,3,1);
	Conv1_combine = CNN_combine_float(Conv1_output,32,3,16);

	Conv2blk1_c3x3b_combine = Convblk(32,16,16,Conv1_combine,conv2blk1_bn_beta,conv2blk1_bn_gamma,conv2blk1_bn_variance,conv2blk1_bn_mean,
	Conv2blk1_c3x3a_float,conv2blk1_stembn_beta,conv2blk1_stembn_gamma,conv2blk1_stembn_variance,conv2blk1_stembn_mean,Conv2blk1_c3x3b_float);


    Conv2blk2_c3x3b_combine = Convblk(32,16,16,Conv2blk1_c3x3b_combine,conv2blk2_bn_beta,conv2blk2_bn_gamma,conv2blk2_bn_variance,conv2blk2_bn_mean,
	Conv2blk2_c3x3a_float,conv2blk2_stembn_beta,conv2blk2_stembn_gamma,conv2blk2_stembn_variance,conv2blk2_stembn_mean,Conv2blk2_c3x3b_float);

    Conv2blk3_c3x3b_combine = Convblk(32,16,16,Conv2blk2_c3x3b_combine,conv2blk3_bn_beta,conv2blk3_bn_gamma,conv2blk3_bn_variance,conv2blk3_bn_mean,
	Conv2blk3_c3x3a_float,conv2blk3_stembn_beta,conv2blk3_stembn_gamma,conv2blk3_stembn_variance,conv2blk3_stembn_mean,Conv2blk3_c3x3b_float);

    Conv3blk1_c3x3b_combine = Convblk_stride2(32,16,32,Conv2blk3_c3x3b_combine,conv3blk1_bn_beta,conv3blk1_bn_gamma,conv3blk1_bn_variance,conv3blk1_bn_mean,
	Conv3blk1_c3x3a_float,conv3blk1_stembn_beta,conv3blk1_stembn_gamma,conv3blk1_stembn_variance,conv3blk1_stembn_mean,Conv3blk1_c3x3b_float);

    Conv3blk2_c3x3b_combine = Convblk(16,32,32,Conv3blk1_c3x3b_combine,conv3blk2_bn_beta,conv3blk2_bn_gamma,conv3blk2_bn_variance,conv3blk2_bn_mean,
	Conv3blk2_c3x3a_float,conv3blk2_stembn_beta,conv3blk2_stembn_gamma,conv3blk2_stembn_variance,conv3blk2_stembn_mean,Conv3blk2_c3x3b_float);

    Conv3blk3_c3x3b_combine = Convblk(16,32,32,Conv3blk2_c3x3b_combine,conv3blk3_bn_beta,conv3blk3_bn_gamma,conv3blk3_bn_variance,conv3blk3_bn_mean,
	Conv3blk3_c3x3a_float,conv3blk3_stembn_beta,conv3blk3_stembn_gamma,conv3blk3_stembn_variance,conv3blk3_stembn_mean,Conv3blk3_c3x3b_float);

    Conv4blk1_c3x3b_combine = Convblk_stride2(16,32,64,Conv3blk3_c3x3b_combine,conv4blk1_bn_beta,conv4blk1_bn_gamma,conv4blk1_bn_variance,conv4blk1_bn_mean,
	Conv4blk1_c3x3a_float,conv4blk1_stembn_beta,conv4blk1_stembn_gamma,conv4blk1_stembn_variance,conv4blk1_stembn_mean,Conv4blk1_c3x3b_float);

    Conv4blk2_c3x3b_combine = Convblk(8,64,64,Conv4blk1_c3x3b_combine,conv4blk2_bn_beta,conv4blk2_bn_gamma,conv4blk2_bn_variance,conv4blk2_bn_mean,
	Conv4blk2_c3x3a_float,conv4blk2_stembn_beta,conv4blk2_stembn_gamma,conv4blk2_stembn_variance,conv4blk2_stembn_mean,Conv4blk2_c3x3b_float);

    Conv4blk3_c3x3b_combine = Convblk(8,64,64,Conv4blk2_c3x3b_combine,conv4blk3_bn_beta,conv4blk3_bn_gamma,conv4blk3_bn_variance,conv4blk3_bn_mean,
	Conv4blk3_c3x3a_float,conv4blk3_stembn_beta,conv4blk3_stembn_gamma,conv4blk3_stembn_variance,conv4blk3_stembn_mean,Conv4blk3_c3x3b_float);


	lastbn = BN_function(Conv4blk3_c3x3b_combine,lastbn_beta,lastbn_gamma,lastbn_variance,lastbn_mean,8,64);
    lastbn = ACT_float(lastbn,8,64);
	lastbn_average = I_AM_SO_HAPPY_BECAUSE_IT_IS_THE_END_AVERGE(lastbn,8,64);
	fc_output = FC_FUNCTION_float(lastbn_average,fct_W,fct_b,64,10);

	//fc_output = softmax(fc_output);

    /*    fprintf(fr,"Conv1_float\n");
    for(int l=0;l<16;l++){
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    fprintf(fr,"%f",Conv1_float[k][j][i][l]);
                    fprintf(fr," ");
                }
                        fprintf(fr,"\n");
            }
                        fprintf(fr,"channel=%d\n",(i+1));
                        fprintf(fr,"\n");
                        fprintf(fr,"\n");
        }
    }

                        fprintf(fr,"\n");
                        fprintf(fr,"\n");
                        fprintf(fr,"\n");

        fprintf(fr,"Conv1_float\n");*/
/*
        fprintf(fr,"input_pad\n");

        for(int i=0;i<16;i++){
            for(int j=0;j<34;j++){
                for(int k=0;k<34;k++){
                    fprintf(fr,"%f",input_pad[k][j][i]);
                    fprintf(fr," ");
                }
                        fprintf(fr,"\n");
            }
                        fprintf(fr,"channel=%d\n",(i+1));
                        fprintf(fr,"\n");
                        fprintf(fr,"\n");
        }

                        fprintf(fr,"\n");
                        fprintf(fr,"\n");
                        fprintf(fr,"\n");

        fprintf(fr,"input_pad\n");

        fprintf(fr,"fc_output\n");

        for(int j=0;j<10;j++){
                fprintf(fr,"fc_output[%d] = %f",j,fc_output[j]);
                fprintf(fr,"\n");
        }

        for(int j=0;j<10;j++){
                printf("classes[%d] = %f",j,classes[j]);
                printf("\n");
        }
*/
/*
        for(int j=0;j<10;j++){
                printf("fc_output[%d] = %f",j,fc_output[j]);
                printf("\n");
        }
*/
	int max = fc_output[0];
	for( int i = 1; i < 10; i++){
		if(fc_output[i] > max){
			outcome = i;
			max = fc_output[i];
		}
	}

	//free_float_3d(input,32);
	free_float_3d(input_minus_mean,32);
	free_float_3d(input_pad,34);
    free_float_4d(Conv1_output,32,3);
    free_float_3d(Conv1_combine,32);
    free_float_3d(Conv2blk1_c3x3b_combine,32);
    free_float_3d(Conv2blk2_c3x3b_combine,32);
    free_float_3d(Conv2blk3_c3x3b_combine,32);
    free_float_3d(Conv3blk1_c3x3b_combine,16);
    free_float_3d(Conv3blk2_c3x3b_combine,16);
    free_float_3d(Conv3blk3_c3x3b_combine,16);
    free_float_3d(Conv4blk1_c3x3b_combine,8);
    free_float_3d(Conv4blk2_c3x3b_combine,8);
    free_float_3d(Conv4blk3_c3x3b_combine,8);

    free_float_3d(lastbn,8);
    free(lastbn_average);
    free(fc_output);


	return outcome;
}

int
main(int argc, char *argv[])
{
    //FILE *fp;
    //float ***IN_put_float;
    //fp = fopen("../ws_resnet/image.txt","r");
    //fp = fopen("../ws_resnet/35683_airplane.txt","r");
    //IN_put_float = Load_input_float(32,3,fp);
    //fclose(fp);
	int outcome;
	load_input();
	//load_all_wt();
	load_all_wt_resnet_float();
/*
	for(int i = 0; i < 64; i++){
        for(int j = 0; j < 10; j++){
            printf("%f ",fct_W[i][j]);
        }
        printf("\n");
	}
*/
/*
    for(int i=0;i<3;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
                printf("%f ",inputs[0][k][j][i]);
            }
            printf("\n");
        }
        printf("\n");
    }
*/

    int error = 0;
    for (int i = 0; i < 5; i++){
        outcome = classify(inputs[i]);
        if(outcome != classes[i])
            error += 1;
        printf("%d %d\n", outcome,i);
    }
    printf("%d\n", error);


    //outcome = classify(IN_put_float);
	return 0;
}
