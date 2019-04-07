#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define num_of_train 50000
#define num_of_data 1461
#define rows_of_data 41
#define ARRAYSIZE(x)  (sizeof(x)/sizeof(*(x)))
void main(){
	double start=omp_get_wtime();

	printf("Preparing Data For Training\n");

// read train data
const char filename[] = "train.csv";
int attributes[num_of_data][rows_of_data];
   FILE *file = fopen(filename, "r");
   if ( file )
   { size_t i, j, k;char buffer[BUFSIZ], *ptr;
      for ( i = 0; fgets(buffer, sizeof buffer, file); ++i )
      {
		  for ( j = 0, ptr = buffer; j < ARRAYSIZE(*attributes); ++j, ++ptr ){
			  attributes[i][j] = (int)strtol(ptr, &ptr, 10);
			}
      }
      fclose(file);
  }


//   for (int i =0;i<num_of_data;i++){
//   	for(int j=0;j<rows_of_data;j++)
//   		{printf("%d ",attributes[i][j]);}
//   	printf("\n");
//   }
// for(int i = 0; i < rows_of_data;i++) printf("%d ", attributes[0][i]);


// read train label
const char cancer_file_name[] = "label.csv";
int classes[num_of_data][1];
   FILE *cancer_file = fopen(cancer_file_name, "r");
   if ( cancer_file )
   { 
      size_t i1, j1, k1;char buffer1[BUFSIZ], *ptr1;
      for ( i1 = 1; fgets(buffer1, sizeof buffer1, cancer_file); ++i1 )
      {for ( j1 = 0, ptr1 = buffer1; j1 < ARRAYSIZE(*classes); ++j1, ++ptr1 )
         {classes[i1][j1] = (int)strtol(ptr1, &ptr1, 10);}
      }
      fclose(cancer_file);}



// for (int i =0;i<num_of_data;i++){
//   	for(int j=0;j<1;j++)
//   		{printf("%d ",classes[i][j]);}
//   	printf("\n");


//   }

printf("Training Begins\n");

// initializing weights
double node1weights[500][rows_of_data],node2weights[500][500],node3weights[500][500],outputlayer[1][500];
printf("Initializing Weights of layer 1\n");
for(int i=0;i<500;i++)for(int j=0;j<rows_of_data;j++)node1weights[i][j]=0.25;
printf("Initializing Weights of layer 2\n");
for(int i=0;i<500;i++)for(int j=0;j<500;j++)node2weights[i][j]=0.25;
printf("Initializing Weights of layer 3\n");
for(int i=0;i<500;i++)for(int j=0;j<500;j++)node3weights[i][j]=0.25;
printf("initializing Weights for outer layer\n");
for(int i=0;i<2;i++)for(int j=0;j<500;j++)outputlayer[i][j]=0.25;

double forpassl1[500],forpassl2[500],forpassl3[500],forpassout[1];
double errl1[500],errl2[500],errl3[500],errout[1];
double errRate = 0.01;

printf("Starting Training\n");
for(int cCount = 0;cCount<num_of_train;cCount++)
{
	int training_sample;
	training_sample = cCount - num_of_data * (cCount / num_of_data);
	{
		//hidden layer 1 forward pass 1
		for(int i=0;i<500;i++){
			forpassl1[i]=0;
			int multifir = 0;
			for(int j=0;j<rows_of_data;j++){
				multifir=multifir+node1weights[i][j]*attributes[training_sample][j];
			}
			forpassl1[i]=1/(1+exp(-multifir));
		}

		//hidden layer 2 forward pass 2
		for(int i=0;i<500;i++){
			forpassl2[i]=0;
			int multifir = 0;
			for(int j=0;j<500;j++){
				multifir=multifir+node2weights[i][j]*forpassl1[j];
			}
			forpassl2[i]=1/(1+exp(-multifir));
		}

		//hidden layer 3 forward pass 3
		for(int i=0;i<500;i++){
			forpassl3[i]=0;
			int multifir = 0;
			for(int j=0;j<500;j++){
				multifir=multifir+node3weights[i][j]*forpassl2[j];
			}
			forpassl3[i]=1/(1+exp(-multifir));
		}

		//output layer forward pass 4
		for(int i=0;i<1;i++){
			forpassout[i]=0;
			int multifir = 0;
			for(int j=0;j<500;j++){
				multifir=multifir+outputlayer[i][j]*forpassl3[j];
			}
			forpassout[i]=1/(1+exp(-multifir));
		}

		//error at output layer 5
		for(int i=0;i<1;i++){
			errout[i]=forpassout[i]*(1-forpassout[i])*(classes[training_sample][i]-forpassout[i]);
		}

		//error at hidden layer 3 6
		for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<2;k++)
				{sum=sum+errout[k]*outputlayer[k][i];}

			errl3[i]=forpassl3[i]*(1-forpassl3[i])*sum;
		}

		//error at hidden layer 2 7
		for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<500;k++)
				{sum=sum+errl3[k]*node3weights[k][i];}

			errl2[i]=forpassl2[i]*(1-forpassl2[i])*sum;
		}

		//error at hidden layer 1 8
		for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<500;k++)
				{sum=sum+errl2[k]*node2weights[k][i];}

			errl1[i]=forpassl1[i]*(1-forpassl1[i])*sum;
		}

		//changing weights in output layer 9
		for(int k=0;k<1;k++)
			for(int j=0;j<500;j++)
			{
				outputlayer[k][j] = outputlayer[k][j] + errRate*(errout[k]*forpassl3[j]);
			}

		// changing weights in hidden layer 3 10
		for(int k=0;k<500;k++)
			for(int j=0;j<500;j++)
			{
				node3weights[k][j] = node3weights[k][j] + errRate*(errl3[k]*forpassl2[j]);
			}

		//changing weights in hidden layer 2 11
		for(int k=0;k<500;k++)
			for(int j=0;j<500;j++)
			{
				node2weights[k][j] = node2weights[k][j] + errRate*(errl2[k]*forpassl1[j]);
			}

		//changing weights in hidden layer 1 12
		for(int k=0;k<500;k++)
			for(int j=0;j< rows_of_data ;j++)
			{
				node1weights[k][j] = node1weights[k][j] + errRate*(errl1[k]*attributes[training_sample][j]);
			}
	}

}
	printf("Training Complete\n");
	double end=omp_get_wtime();
	printf("%lf\n", (end-start));
}













// #include <string>
// #include <vector>
// #include <sstream> //istringstream
// #include <iostream> // cout
// #include <fstream> // ifstream
 
// using namespace std;
 
// /**
//  * Reads csv file into table, exported as a vector of vector of doubles.
//  * @param inputFileName input file name (full path).
//  * @return data as vector of vector of doubles.
//  */
// vector<vector<double> > parse2DCsvFile(string inputFileName) {
 
//     vector<vector<double> > data;
//     ifstream inputFile(inputFileName);
//     int l = 0;
 
//     while (inputFile) {
//         l++;
//         string s;
//         if (!getline(inputFile, s)) break;
//         if (s[0] != '#') {
//             istringstream ss(s);
//             vector<double> record;
 
//             while (ss) {
//                 string line;
//                 if (!getline(ss, line, ','))
//                     break;
//                 try {
//                     record.push_back(stof(line));
//                 }
//                 catch (const std::invalid_argument e) {
//                     cout << "NaN found in file " << inputFileName << " line " << l
//                          << endl;
//                     e.what();
//                 }
//             }
 
//             data.push_back(record);
//         }
//     }
 
//     if (!inputFile.eof()) {
//         cerr << "Could not read file " << inputFileName << "\n";
//         __throw_invalid_argument("File not found.");
//     }
 
//     return data;
// }
 
// int main()
// {   
    
//     std::vector<vector<double> > data = parse2DCsvFile("train.csv");
//     std::vector<vector<double> > label = parse2DCsvFile("label.csv");
//     // for (auto l : data) {
//     //     for (auto x : l)
//     //         cout << x << " ";
//     //     cout << endl;
//     // }
//     auto v = data;
//     std::copy(begin(v), end(v), std::ostream_iterator<int>(std::cout, " "));
//     // cout << "first element from begin(): " << *label.begin() << endl;
//     // for( std::vector<vector<double> >::const_iterator i=data.begin(); i!=data.end(); ++i)
//     //     std::cout << *i << ' ';
//     // cout<< label[23]<<endl;
//     return 0;
// }