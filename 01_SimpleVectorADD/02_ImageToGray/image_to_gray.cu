
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <direct.h>  
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
using namespace std;

struct Pixel {
	unsigned char r, g, b, a;
};

void ConvertToGrayCPU(unsigned char* imageData, int width, int height) {
	// Convert Image to Gray
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Pointer to a Pixel type, used to access pixel data
			Pixel* ptr_pixel =
				// Calculate the address of the pixel at (x, y)
				(Pixel*)(imageData +
					// Calculate the linear index in the image data array
					(y * width + x) * 4 // Each pixel is 4 bytes (RGBA)
					);

			float pixelValue = (unsigned char)(ptr_pixel->r * 0.2126f + ptr_pixel->g * 0.7152f + ptr_pixel->b * 0.0722f);
			ptr_pixel->r = pixelValue;
			ptr_pixel->g = pixelValue;
			ptr_pixel->b = pixelValue;
			ptr_pixel->a = 255;

		}
	}
}

__global__ void ConvertToGrayGPU(unsigned char* imageData) {
	// Calculate the global x coordinate of the thread
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the global y coordinate of the thread
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the linear index in the image data array based on (x, y)
	uint32_t idx = y * (blockDim.x * gridDim.x) + x;
	Pixel* ptr_pixel = &((Pixel*)imageData)[idx];
	float pixelValue = ptr_pixel->r * 0.2126f + ptr_pixel->g * 0.7152f + ptr_pixel->b * 0.0722f;

	// Set grayscale values
	ptr_pixel->r = pixelValue;
	ptr_pixel->g = pixelValue;
	ptr_pixel->b = pixelValue;
	ptr_pixel->a = 255; 
}


int main(int argc, char** argv)
{
	//Open Image
	int width, height, componentCount;
	// Load Image
	unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 0);
	if (!imageData) {
		std::cerr << "Image not found\""<<argv[1] << "\"";
		free(imageData);
		return -1;
	}

	// Validate image dimensions
	if (width % 32 || height % 32) {
		std::cerr << "Image dimensions must be multiple of 32";
		free(imageData);
		return -1;
	}

	// Convert Image to Gray
	//ConvertToGrayCPU(imageData, width, height);

	// Memory Allocation on GPU
	unsigned char* d_imageDataGpu = nullptr;
	cudaMalloc(&d_imageDataGpu, width * height * 4);
	cudaMemcpy(d_imageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice); //Copy Data to GPU


	// Process On GPU
	cout << "Processing on GPU" << endl;\
	dim3 blockSize(32, 32);
	dim3 gridSize(width / blockSize.x, height / blockSize.y);

	ConvertToGrayGPU << <gridSize, blockSize >> > (d_imageDataGpu);
	cout << "Done Processing on GPU" << endl;
	cout << "Copying Data back to CPU" << endl;

	//Copy Data back to CPU
	cudaMemcpy(imageData, d_imageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);


	// Build output filename
	std::string outputFilename = (argv[1]);
	outputFilename = outputFilename.substr(0, outputFilename.find_last_of('.')) + "_gray.png";

	//Write Image back to disk
	cout << "Writing Gray Image to disk: " << endl;
	stbi_write_png(outputFilename.c_str(), width, height, 4, imageData, 4*width);

	cout << "Done!! " << endl;

	//Free Image Data
	stbi_image_free(imageData);
	cudaFree(d_imageDataGpu);
}