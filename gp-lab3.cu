#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
//#include "../lib/cuPrintf.cu"

using namespace std;

const uint32_t BLOCK_DIM = 32;
const uint32_t MAX_CLASS_COUNT = 32;
//const uint32_t MAX_CLASS_ELEMENTS_COUNT = 524288;

namespace Pixel {
	//const uint32_t ELEMENTS_CNT = 4;
	const uint32_t RED = 8 * 0;
	const uint32_t GREEN = 8 * 1;
	const uint32_t BLUE = 8 * 2;
	const uint32_t ALPHA = 8 * 3;
}


/*
==========
STRUCTURES
==========
*/

struct Position {
	int32_t X;
	int32_t Y;
};
struct ModifiedPixel {
	float Red;
	float Green;
	float Blue;
};

texture<uint32_t, 2, cudaReadModeElementType> OriginalImage;
//__constant__ Position class_map[MAX_CLASS_COUNT * MAX_CLASS_ELEMENTS_COUNT];
__constant__ uint8_t ClassCount[1];
__constant__ uint32_t ClassMapElementsCounts[MAX_CLASS_COUNT];
__constant__ uint32_t ClassMapElementsOffsets[MAX_CLASS_COUNT];
__constant__ ModifiedPixel ClassAVG[MAX_CLASS_COUNT];
texture<Position, 1, cudaReadModeElementType> ClassMap;

/*
===========
DEVICE-HOST
===========
*/

/*__device__ double GetIntensity(Pixel pixel) {
	return (.3 * (double) pixel.Red) + (.59 * (double) pixel.Green) + (.11 * (double) pixel.Blue);
}*/



__host__ __device__ ModifiedPixel SetModifiedPixel() {
	ModifiedPixel res;
	res.Red = 0.;
	res.Green = 0.;
	res.Blue = 0.;
	return res;
}

__host__ __device__ ModifiedPixel SetModifiedPixel(float r, float g, float b) {
	ModifiedPixel res;
	res.Red = r;
	res.Green = g;
	res.Blue = b;
	return res;
}
__host__ __device__ ModifiedPixel SetModifiedPixel(uint8_t r, uint8_t g, uint8_t b) {
	ModifiedPixel res;
	res.Red = (float) r;
	res.Green = (float) g;
	res.Blue = (float) b;
	return res;
}

__device__ __host__ bool IsCorrectPos(Position pos, uint32_t height, uint32_t width) {
	return (pos.X >= 0 && pos.Y >= 0 && pos.X < (int32_t) width && pos.Y < (int32_t) height);
}

__device__ __host__ int32_t GetLinearizedPosition(Position pos, uint32_t height, uint32_t width) {
	return (IsCorrectPos(pos, height, width)) ? (pos.Y * (int32_t) width + pos.X) : -1;
}
__device__ __host__ uint32_t MakePixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha) {
	return ((uint32_t) red << Pixel::RED) + ((uint32_t) green << Pixel::GREEN) +
			((uint32_t) blue << Pixel::BLUE) + ((uint32_t) alpha << Pixel::ALPHA);
}

__device__ __host__ uint8_t GetPixelElement(uint32_t pixel, uint32_t element) {
	return (uint8_t) (pixel >> element) & 255;
}

__device__ __host__ uint32_t SetPixelElement(uint32_t pixel, uint32_t element, uint8_t value) {
	return (~(255 << element) & pixel) + (((uint32_t) value) << element);
	//cout << element << endl;
	//return 255 >> element;
}

__device__ __host__ void PixelSumm(ModifiedPixel &a, ModifiedPixel b) {
	//return ModifiedPixel(a.Red + b.Red, a.Green + b.Green, a.Blue + b.Blue);
	a.Red += b.Red;
	a.Green += b.Green;
	a.Blue += b.Blue;
}
__device__ __host__ void PixelDiff(ModifiedPixel &a, ModifiedPixel b) {
	//return ModifiedPixel(a.Red - b.Red, a.Green - b.Green, a.Blue - b.Blue);
	a.Red -= b.Red;
	a.Green -= b.Green;
	a.Blue -= b.Blue;
}
__device__ __host__ float PixelMult(ModifiedPixel a, ModifiedPixel b) {
	//return ModifiedPixel(a.Red * b.Red, a.Green * b.Green, a.Blue * b.Blue);
	return (a.Red * b.Red) + (a.Green * b.Green) + (a.Blue * b.Blue);
}
__device__ __host__ void PixelMultNum(ModifiedPixel &a, float num) {
	//return ModifiedPixel(a.Red * num, a.Green * num, a.Blue * num);
	a.Red *= num;
	a.Green *= num;
	a.Blue *= num;
}
__device__ __host__ ModifiedPixel PixelMinus(ModifiedPixel a) {
	return SetModifiedPixel(-a.Red, -a.Green, -a.Blue);
}
__device__ __host__ ModifiedPixel ConvertPixelToModified(uint32_t pixel) {
	return SetModifiedPixel(GetPixelElement(pixel, Pixel::RED), GetPixelElement(pixel, Pixel::GREEN),
		GetPixelElement(pixel, Pixel::BLUE));
}
__device__ __host__ uint32_t ConvertPixelFromModified(ModifiedPixel mod_pixel) {
	return MakePixel((uint8_t) mod_pixel.Red, (uint8_t) mod_pixel.Green, (uint8_t) mod_pixel.Blue, 0);
}

/*
======
DEVICE
======
*/

__device__ float GetMinDist(Position pos, uint8_t j) {
	ModifiedPixel px_bas = ConvertPixelToModified(tex2D(OriginalImage, pos.X, pos.Y));
	//cuPrintf("%f:%f:%f\n", px_bas.Red, px_bas.Green, px_bas.Blue);
	//cuPrintf("%f:%f:%f\n", ClassAVG[j].Red, ClassAVG[j].Green, ClassAVG[j].Blue);

	PixelDiff(px_bas, ClassAVG[j]);
	//cuPrintf("%f:%f:%f\n", px_bas.Red, px_bas.Green, px_bas.Blue);
	return PixelMult(px_bas, PixelMinus(px_bas));
}

__device__ void SetClass(Position pos, uint32_t *map_out, uint32_t height, uint32_t width) {
	uint8_t class_number = 0;
	float max_val = 0.;
	uint8_t is_defined = 0;
	//cuPrintf("ClassCount = %d\n", (uint32_t) ClassCount[0]);

	for (uint8_t j = 0; j < ClassCount[0]; j++) {
		if (!is_defined) {
			max_val = GetMinDist(pos, j);
			class_number = j;
			is_defined = 1;
			continue;
		}

		//cuPrintf("j: %d\nMax: %f\nCurr: %f\n", j, max_val, GetMinDist(pos, j));

		if (GetMinDist(pos, j) > max_val) {
			max_val = GetMinDist(pos, j);
			class_number = j;
		}
	}

	//Class calculating

	map_out[GetLinearizedPosition(pos, height, width)] = tex2D(OriginalImage, pos.X, pos.Y);
	map_out[GetLinearizedPosition(pos, height, width)] = SetPixelElement(
		map_out[GetLinearizedPosition(pos, height, width)], Pixel::ALPHA, class_number);
}

/*
======
GLOBAL
======
*/

__global__ void Classificator(uint32_t height, uint32_t width, uint32_t *map_out) {

	Position start, offset;
	start.X = blockIdx.x * blockDim.x + threadIdx.x;
	start.Y = blockIdx.y * blockDim.y + threadIdx.y;

	offset.X = gridDim.x * blockDim.x;
	offset.Y = gridDim.y * blockDim.y;

	Position pos;
	for (pos.X = start.X; pos.X < width; pos.X += offset.X) {
		for (pos.Y = start.Y; pos.Y < height; pos.Y += offset.Y) {
			if (pos.X < width && pos.Y < height) {
				//cuPrintf("\n%d:%d\n", pos.X, pos.Y);
				SetClass(pos, map_out, height, width);
			}
		}
	}
}

/*
====
HOST
====
*/

__host__ void InitPixelMap(uint32_t **pixel, uint32_t height, uint32_t width) {
	*pixel = new uint32_t[height * width];
}

__host__ void DestroyPixelMap(uint32_t **pixel) {
	delete [] (*pixel);
	*pixel = NULL;
}	

__host__ void ReadImageFromFile(uint32_t **pixel, uint32_t *height, uint32_t *width,
		string filename) {
	FILE *file = fopen(filename.c_str(), "rb");
	uint32_t sizes[2];
	fread(sizes, sizeof(uint32_t), 2, file);
	*width = sizes[0];
	*height = sizes[1];

	uint32_t size = (*height) * (*width);

	InitPixelMap(pixel, *height, *width);
	fread(*pixel, sizeof(uint32_t), size, file);
	fclose(file);
}

__host__ void WriteImageToFile(uint32_t *pixel, uint32_t height, uint32_t width, string filename) {
	FILE *file = fopen(filename.c_str(), "wb");
	uint32_t sizes[2] = {width, height};
	fwrite(sizes, sizeof(uint32_t), 2, file);

	uint32_t size = height * width;
	fwrite(pixel, sizeof(uint32_t), size, file);
	fclose(file);
}

__host__ void FileGeneratorTest() {
	uint32_t *pixel;
	uint32_t height = 3;
	uint32_t width = 3;
	InitPixelMap(&pixel, height, width);

	string filename = "in.data";
	pixel[0] = MakePixel(162, 223, 76, 0);
	pixel[1] = MakePixel(247, 201, 254, 0);
	pixel[2] = MakePixel(158, 216, 69, 0);

	pixel[3] = MakePixel(180, 232, 83, 0);
	pixel[4] = MakePixel(153, 209, 77, 0);
	pixel[5] = MakePixel(146, 221, 86, 0);

	pixel[6] = MakePixel(169, 224, 76, 0);
	pixel[7] = MakePixel(247, 209, 250, 0);
	pixel[8] = MakePixel(212, 208, 233, 0);

	WriteImageToFile(pixel, height, width, filename);
	DestroyPixelMap(&pixel);
}
__host__ uint32_t GetRandomPixel() {
	return MakePixel(rand() % 256, rand() % 256, rand() % 256, 0);
}
__host__ void FileGenerator(string filename) {
	uint32_t *pixel;
	uint32_t height = 400;
	uint32_t width = 640;
	InitPixelMap(&pixel, height, width);

	for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			pixel[i * width + j] = GetRandomPixel();
		}
	}
	WriteImageToFile(pixel, height, width, filename);
	DestroyPixelMap(&pixel);
}
__host__ void FileGeneratorBig(uint32_t height, uint32_t width, string filename) {
	uint32_t *pixel;
	InitPixelMap(&pixel, height, width);

	for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			uint8_t curr;
			if (i == 0 || j == 0 || i == height - 1 || j == width - 1) {
				curr = 1;
			} else {
				curr = 3;
			}
			pixel[i * width + j] = MakePixel(curr, curr, curr, 0);
		}
	}

	WriteImageToFile(pixel, height, width, filename);
	DestroyPixelMap(&pixel);
}

__host__ ModifiedPixel GetAVG(uint8_t j, uint32_t *map_in, uint32_t height, uint32_t width,
		vector <Position> &class_elements, uint32_t *class_elements_counts,
		uint32_t *class_elements_offsets) {
	ModifiedPixel res = SetModifiedPixel();
	//printf("%f:%f:%f\n", res.Red, res.Green, res.Blue);


	for (uint32_t i = class_elements_offsets[j];
			i < class_elements_offsets[j] + class_elements_counts[j]; i++) {
		//PixelSumm(res, ConvertPixelToModified(tex2D(OriginalImage, tex1D(ClassMap, i).X, tex1D(ClassMap, i).Y)));
		PixelSumm(res, ConvertPixelToModified(map_in[GetLinearizedPosition(class_elements[i], height, width)]));
	
	}
	//printf("%f:%f:%f\n", res.Red, res.Green, res.Blue);
	PixelMultNum(res, 1./((float) class_elements_counts[j]));
	//printf("%f:%f:%f\n", res.Red, res.Green, res.Blue);
	return res;
}

__host__ int main(void) {
	//cout << "INIT" << endl;
	srand(time(NULL));
	//FileGeneratorBig(100, 100, "inbig.data");
	//FileGenerator("inrand.data");
	//FileGeneratorTest();

	string file_in, file_out;
	cin >> file_in >> file_out;
	//cout << "FILE INIT COMPLETED" << endl;

	//FileGenerator();
	uint32_t *pixel_in;
	uint32_t *pixel_out;
	uint32_t height, width;

	ReadImageFromFile(&pixel_in, &height, &width, file_in);
	//cout << "READ IMAGE COMPLETED" << endl;

	InitPixelMap(&pixel_out, height, width);

	uint32_t class_elements_counts[MAX_CLASS_COUNT];
	uint32_t class_elements_offsets[MAX_CLASS_COUNT];
	ModifiedPixel class_elements_avg[MAX_CLASS_COUNT];

	vector <Position> class_elements(0);
	uint32_t cnt = 0;

	//cout << "INPUT CLASSES" << endl;

	uint8_t class_count;
	uint32_t tmp;
	cin >> tmp;
	class_count = (uint8_t) tmp;
	//cout << class_count << endl;
	for (uint8_t i = 0; i < class_count; i++) {
		//cout << "CL_CNT = " << class_count << endl;
		//cout << "i = " << (uint32_t) i << endl;
		//cout << (uint32_t) i << " < " << class_count << " = ";
		//cout << (i < class_count ? "true" : "false") << endl;
		cin >> class_elements_counts[i];
		//cout << class_elements_counts[i] << endl;
		class_elements_offsets[i] = cnt;
		cnt += class_elements_counts[i];
		for (uint32_t j = 0; j < class_elements_counts[i]; j++) {
			//cout << "\t" << (uint32_t) j << endl;
			Position tmp_pos;
			cin >> tmp_pos.X >> tmp_pos.Y;
			class_elements.push_back(tmp_pos);
		}
		class_elements_avg[i] = GetAVG(i, pixel_in, height, width, class_elements,
			class_elements_counts, class_elements_offsets);
		//printf("%f:%f:%f\n", class_elements_avg[i].Red, class_elements_avg[i].Green, class_elements_avg[i].Blue);

	}
	//cout << "INPUT END" << endl;

	uint32_t *cuda_pixel_out;

	//Texture init begin
	cudaArray *cuda_pixel_in;
	cudaChannelFormatDesc ch1 = cudaCreateChannelDesc<uint32_t>();
	cudaMallocArray(&cuda_pixel_in, &ch1, width, height);
	cudaMemcpyToArray(cuda_pixel_in, 0, 0, pixel_in, sizeof(uint32_t) * height * width, cudaMemcpyHostToDevice);
	
	OriginalImage.addressMode[0] = cudaAddressModeClamp;
	OriginalImage.addressMode[1] = cudaAddressModeClamp;

	OriginalImage.channelDesc = ch1;
	OriginalImage.filterMode = cudaFilterModePoint;
	OriginalImage.normalized = false;
	cudaBindTextureToArray(OriginalImage, cuda_pixel_in, ch1);
	//Texture init end

	//Texture init begin
	cudaArray *cuda_classmap;
	cudaChannelFormatDesc ch2 = cudaCreateChannelDesc<Position>();
	cudaMallocArray(&cuda_classmap, &ch2, width, height);
	cudaMemcpyToArray(cuda_classmap, 0, 0, pixel_in, sizeof(uint32_t) * height * width, cudaMemcpyHostToDevice);
	
	ClassMap.addressMode[0] = cudaAddressModeClamp;
	ClassMap.addressMode[1] = cudaAddressModeClamp;

	ClassMap.channelDesc = ch2;
	ClassMap.filterMode = cudaFilterModePoint;
	ClassMap.normalized = false;
	cudaBindTextureToArray(ClassMap, cuda_classmap, ch2);
	//Texture init end

	cudaMemcpyToSymbol(ClassMapElementsCounts, class_elements_counts, sizeof(uint32_t) * class_count);
	cudaMemcpyToSymbol(ClassMapElementsOffsets, class_elements_offsets, sizeof(uint32_t) * class_count);
	cudaMemcpyToSymbol(ClassCount, &class_count, sizeof(uint8_t));
	cudaMemcpyToSymbol(ClassAVG, class_elements_avg, sizeof(ModifiedPixel) * class_count);
	cudaMalloc((void**) &cuda_pixel_out, sizeof(uint32_t) * width * height);

	dim3 threads_per_block(width, height);
	dim3 blocks_per_grid(1, 1);

	if (height * width > BLOCK_DIM * BLOCK_DIM){
		threads_per_block.x = BLOCK_DIM;
		threads_per_block.y = BLOCK_DIM;
		blocks_per_grid.x = ceil((double) (width) / (double)(threads_per_block.x));
		blocks_per_grid.y = ceil((double) (height) / (double)(threads_per_block.y));
	}

	//cudaPrintfInit();
	Classificator<<<blocks_per_grid, threads_per_block>>>(height, width, cuda_pixel_out);
	//cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(pixel_out, cuda_pixel_out, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost);

	cudaEventDestroy(syncEvent);

	cudaUnbindTexture(OriginalImage);
	cudaFreeArray(cuda_pixel_in);
	cudaFree(cuda_pixel_out);

	WriteImageToFile(pixel_out, height, width, file_out);

	DestroyPixelMap(&pixel_in);
	DestroyPixelMap(&pixel_out);

	return 0;
}

